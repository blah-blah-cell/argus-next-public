//! Per-user Windows setup program for the Argus Next public prototype.
//!
//! The payload is appended by `argus-setup-builder`. Every file is verified
//! against its recorded SHA-256 before anything is written, so a truncated or
//! altered download fails instead of installing partially.

use argus_setup::container::{sanitize_relative, Package};
use argus_setup::ownership::{fresh_install_preflight, hash_file, write_record, Removal};
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const APP_EXE: &str = "argus-desktop.exe";
const APP_NAME: &str = "Argus Next Public";
const UNINSTALL_KEY: &str =
    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\ArgusNextPublic";

fn main() {
    if let Err(error) = run() {
        eprintln!("\nSetup failed: {error}");
        std::process::exit(1);
    }
}

#[derive(Default)]
struct Options {
    install_dir: Option<PathBuf>,
    silent: bool,
    no_launch: bool,
    uninstall: bool,
    from_temp: bool,
    verify_only: bool,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = Options::default();
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--silent" | "-s" => options.silent = true,
            "--no-launch" => options.no_launch = true,
            "--uninstall" => options.uninstall = true,
            "--from-temp" => options.from_temp = true,
            "--verify" => options.verify_only = true,
            "--install-dir" => {
                let value = arguments.next().ok_or("--install-dir requires a path")?;
                options.install_dir = Some(std::path::absolute(value)?);
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}\nTry --help.").into()),
        }
    }

    let setup = env::current_exe()?;
    // Uninstall never needs the embedded payload: the same stub is copied into
    // the install directory so removal survives deleting the downloaded setup.
    if options.uninstall {
        return uninstall(&setup, &options);
    }

    let mut package = Package::open(&setup)?;
    let version = package
        .read_text("VERSION.txt")?
        .unwrap_or_else(|| "unknown".to_owned())
        .trim()
        .to_owned();

    if options.verify_only {
        package.verify()?;
        println!(
            "Payload verified: {} files, version {version}.",
            package.entries.len()
        );
        return Ok(());
    }

    package.verify()?;
    install(&setup, &mut package, &options, &version)
}

fn install(
    setup: &Path,
    package: &mut Package,
    options: &Options,
    version: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if app_running()? {
        return Err(format!("{APP_NAME} is running. Close it and run setup again.").into());
    }
    let directory = match &options.install_dir {
        Some(path) => path.clone(),
        None => default_install_directory()?,
    };

    fresh_install_preflight(&directory, shared_installation_exists()?)?;

    println!("{APP_NAME} {version}");
    println!("Install location: {}", directory.display());
    if !options.silent {
        println!("This installs for the current user and needs no administrator rights.");
        if !confirm("Continue? [Y/n] ", true) {
            println!("Cancelled.");
            return Ok(());
        }
    }

    // Repeat after any prompt; never extract over a pre-existing installation.
    fresh_install_preflight(&directory, shared_installation_exists()?)?;
    fs::create_dir_all(
        directory
            .parent()
            .ok_or("install directory has no parent")?,
    )?;
    fs::create_dir(&directory)?;
    package.extract(|relative| Ok(directory.join(sanitize_relative(relative)?)))?;
    write_install_record(&directory, version, package)?;
    // Copy only the stub, not the 21 MB payload, so the installed uninstaller
    // stays small while still working after the download is deleted.
    let uninstaller = directory.join("argus-uninstall.exe");
    copy_prefix(setup, &uninstaller, package.stub_length())?;
    let mut owned = package
        .entries
        .iter()
        .map(|entry| (entry.path.clone(), argus_setup::sha256::hex(&entry.hash)))
        .collect::<Vec<_>>();
    owned.push((
        "INSTALL.txt".to_owned(),
        hash_file(&directory.join("INSTALL.txt"))?,
    ));
    write_record(&directory, &owned)?;
    if shared_installation_exists()? {
        return Err(
            "Argus registration appeared during installation; refusing to overwrite it".into(),
        );
    }
    create_shortcuts(&directory)?;
    write_uninstall_entry(&uninstaller, &directory, version)?;

    println!("Installed {} files.", package.entries.len());
    println!("A Start menu shortcut and a desktop shortcut were created.");
    println!("Capture stays stopped until you start it inside the app.");

    if !options.no_launch && !options.silent && confirm("Start Argus Next now? [Y/n] ", true) {
        launch(&directory.join(APP_EXE), &directory)?;
    }
    Ok(())
}

fn uninstall(setup: &Path, options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let directory = match &options.install_dir {
        Some(path) => path.clone(),
        None => default_install_directory()?,
    };

    // Validate before spawning, prompting, changing registration, or deleting.
    // Legacy installations without this record require manual removal.
    let removal = Removal::validate(&directory, setup)?;

    // The uninstaller usually lives inside the directory it must delete.
    if !options.from_temp {
        let temporary =
            env::temp_dir().join(format!("argus-next-uninstall-{}.exe", std::process::id()));
        fs::copy(setup, &temporary)?;
        let mut command = Command::new(&temporary);
        command.arg("--uninstall").arg("--from-temp");
        if options.silent {
            command.arg("--silent");
        }
        if let Some(path) = &options.install_dir {
            command.arg("--install-dir").arg(path);
        }
        command.spawn()?;
        println!("Uninstall continues in the background.");
        return Ok(());
    }

    if app_running()? {
        return Err(format!("{APP_NAME} is running. Close it and try again.").into());
    }

    if !removal.remove()? {
        println!("Kept user-added files in {}", directory.display());
    }
    remove_shortcuts(&directory)?;
    remove_uninstall_entry(&directory)?;

    let data = default_data_directory()?;
    println!("Kept data: {}", data.display());
    println!("{APP_NAME} uninstalled.");
    Ok(())
}

fn write_install_record(directory: &Path, version: &str, package: &Package) -> io::Result<()> {
    let installed_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let mut text = format!(
        "{APP_NAME} {version}\n\
         Installed by argus-setup from its verified payload.\n\
         Installed at (unix seconds): {installed_at}\n\
         Files: {}\n\n",
        package.entries.len()
    );
    for entry in &package.entries {
        text.push_str(&format!(
            "{}  {}\n",
            argus_setup::sha256::hex(&entry.hash),
            entry.path
        ));
    }
    fs::write(directory.join("INSTALL.txt"), text)
}

fn powershell_quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn shared_installation_script() -> &'static str {
    // Read-only, errors fail closed. Use the same Windows known folders as the
    // shortcut writer, rather than guessing from an unredirected Desktop path.
    "$ErrorActionPreference='Stop'; \
        $paths=@('HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\ArgusNextPublic'); \
        foreach ($folder in @('Programs','Desktop')) { \
            $base=[Environment]::GetFolderPath($folder); \
            if ($base) { $paths += (Join-Path $base 'Argus Next Public.lnk') } \
        }; \
        $found=$false; foreach ($path in $paths) { \
            if (Test-Path -LiteralPath $path -ErrorAction Stop) { $found=$true } \
        }; [Console]::Write($found.ToString())"
}

fn shared_installation_exists() -> io::Result<bool> {
    let script = shared_installation_script();
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other("cannot verify shared Argus registration"));
    }
    match String::from_utf8_lossy(&output.stdout).trim() {
        "True" => Ok(true),
        "False" => Ok(false),
        _ => Err(io::Error::other(
            "invalid shared registration probe response",
        )),
    }
}

fn create_shortcuts(directory: &Path) -> io::Result<()> {
    let appdata = env::var_os("APPDATA")
        .ok_or_else(|| io::Error::other("APPDATA is not set; cannot create shortcuts"))?;
    let start_menu = PathBuf::from(appdata).join(r"Microsoft\Windows\Start Menu\Programs");
    fs::create_dir_all(&start_menu)?;

    let link = start_menu.join("Argus Next Public.lnk");
    let target = directory.join(APP_EXE);
    let script = format!(
        "$ErrorActionPreference='Stop'; \
         $shell=New-Object -ComObject WScript.Shell; \
         $target='{target}'; $work='{work}'; \
         $links=@('{start}'); \
         $desktop=[Environment]::GetFolderPath('Desktop'); \
         if ($desktop) {{ $links += (Join-Path $desktop 'Argus Next Public.lnk') }}; \
         foreach ($link in $links) {{ \
            $shortcut=$shell.CreateShortcut($link); \
            $shortcut.TargetPath=$target; \
            $shortcut.WorkingDirectory=$work; \
            $shortcut.IconLocation=($target + ',0'); \
            $shortcut.Description='Argus Next Public'; \
            $shortcut.Save() \
         }}",
        target = powershell_quote(&target),
        work = powershell_quote(directory),
        start = powershell_quote(&link),
    );

    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "could not create shortcuts: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

fn shortcut_cleanup_script(directory: &Path) -> String {
    format!(
        "$ErrorActionPreference='Stop'; $shell=New-Object -ComObject WScript.Shell; \
         $expected=[IO.Path]::GetFullPath('{}'); \
         foreach ($folder in @('Programs','Desktop')) {{ \
           $base=[Environment]::GetFolderPath($folder); if (!$base) {{ continue }}; \
           $link=Join-Path $base 'Argus Next Public.lnk'; \
           if (Test-Path -LiteralPath $link) {{ \
             $item=Get-Item -LiteralPath $link; \
             if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {{ continue }}; \
             $target=$shell.CreateShortcut($link).TargetPath; \
             if ($target -and ([IO.Path]::GetFullPath($target) -ieq $expected)) {{ \
               Remove-Item -LiteralPath $link -Force \
             }} \
           }} \
         }}",
        powershell_quote(&directory.join(APP_EXE))
    )
}

fn remove_shortcuts(directory: &Path) -> io::Result<()> {
    let script = shortcut_cleanup_script(directory);
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ])
        .output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(io::Error::other("could not clean owned shortcuts"))
    }
}

fn write_uninstall_entry(uninstaller: &Path, directory: &Path, version: &str) -> io::Result<()> {
    let icon = directory.join(APP_EXE);
    let setup_path = uninstaller.to_string_lossy().to_string();
    let size_kb = directory_size(directory)? / 1024;
    let values = [
        ("DisplayName", APP_NAME.to_owned(), "REG_SZ"),
        ("DisplayVersion", version.to_owned(), "REG_SZ"),
        ("Publisher", "Argus Next".to_owned(), "REG_SZ"),
        (
            "InstallLocation",
            directory.to_string_lossy().to_string(),
            "REG_SZ",
        ),
        ("DisplayIcon", icon.to_string_lossy().to_string(), "REG_SZ"),
        (
            "UninstallString",
            uninstall_command(&setup_path, directory, false),
            "REG_SZ",
        ),
        (
            "QuietUninstallString",
            uninstall_command(&setup_path, directory, true),
            "REG_SZ",
        ),
        ("EstimatedSize", size_kb.to_string(), "REG_DWORD"),
    ];
    for (name, value, kind) in values {
        reg(&[
            "add",
            UNINSTALL_KEY,
            "/f",
            "/v",
            name,
            "/t",
            kind,
            "/d",
            &value,
        ])?;
    }
    reg(&[
        "add",
        UNINSTALL_KEY,
        "/f",
        "/v",
        "NoModify",
        "/t",
        "REG_DWORD",
        "/d",
        "1",
    ])?;
    reg(&[
        "add",
        UNINSTALL_KEY,
        "/f",
        "/v",
        "NoRepair",
        "/t",
        "REG_DWORD",
        "/d",
        "1",
    ])
}

fn registration_cleanup_script(directory: &Path) -> String {
    format!(
        "$ErrorActionPreference='Stop'; \
         $key='HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\ArgusNextPublic'; \
         if (Test-Path -LiteralPath $key) {{ \
           $current=Get-ItemProperty -LiteralPath $key; \
           if ($current.InstallLocation -and \
               ([IO.Path]::GetFullPath($current.InstallLocation) -ieq [IO.Path]::GetFullPath('{}'))) {{ \
             Remove-Item -LiteralPath $key \
           }} \
         }}",
        powershell_quote(directory)
    )
}

fn remove_uninstall_entry(directory: &Path) -> io::Result<()> {
    let script = registration_cleanup_script(directory);
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(io::Error::other(
            "could not clean owned uninstall registration",
        ))
    }
}

fn uninstall_command(executable: &str, directory: &Path, silent: bool) -> String {
    // The registry command is launched independently of this process, so it
    // must carry a custom install location through to the temporary uninstaller.
    let directory = directory.to_string_lossy();
    // Windows argument parsing treats a backslash before the closing quote as
    // an escape. Double only trailing backslashes (including a drive root).
    let trailing_slashes = directory.chars().rev().take_while(|c| *c == '\\').count();
    format!(
        "\"{executable}\" --uninstall --install-dir \"{}{}\"{}",
        directory,
        "\\".repeat(trailing_slashes),
        if silent { " --silent" } else { "" }
    )
}

fn reg(arguments: &[&str]) -> io::Result<()> {
    let output = Command::new("reg").args(arguments).output()?;
    if output.status.success() {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "registry command failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    )))
}

fn copy_prefix(source: &Path, destination: &Path, length: u64) -> io::Result<()> {
    let input = fs::File::open(source)?;
    let mut output = fs::File::create(destination)?;
    let copied = io::copy(&mut input.take(length), &mut output)?;
    if copied != length {
        return Err(io::Error::other(
            "the setup executable changed while installing; run setup again",
        ));
    }
    output.flush()
}

fn directory_size(directory: &Path) -> io::Result<u64> {
    let mut total = 0_u64;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += directory_size(&entry.path())?;
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}

fn app_running() -> io::Result<bool> {
    let output = Command::new("tasklist")
        .args(["/FI", "IMAGENAME eq argus-desktop.exe", "/NH"])
        .output()?;
    let report = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
    Ok(report.contains("argus-desktop.exe"))
}

fn confirm(prompt: &str, default_yes: bool) -> bool {
    print!("{prompt}");
    let _ = io::stdout().flush();
    let mut line = String::new();
    if io::stdin().read_line(&mut line).is_err() {
        return default_yes;
    }
    match line.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default_yes,
    }
}

fn launch(executable: &Path, directory: &Path) -> io::Result<()> {
    Command::new(executable)
        .current_dir(directory)
        .spawn()
        .map(|_| ())
}

fn default_install_directory() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let local = env::var_os("LOCALAPPDATA").ok_or("LOCALAPPDATA is not set")?;
    Ok(PathBuf::from(local)
        .join("Programs")
        .join("ArgusNextPublic"))
}

fn default_data_directory() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let local = env::var_os("LOCALAPPDATA").ok_or("LOCALAPPDATA is not set")?;
    Ok(PathBuf::from(local).join("ArgusNextPublic"))
}

fn print_help() {
    println!(
        "{APP_NAME} setup\n\n\
         Usage:\n  \
           argus-setup [--silent] [--no-launch] [--install-dir <path>]\n  \
           argus-setup --uninstall [--silent]\n  \
           argus-setup --verify\n\n\
         Options:\n  \
           --silent        install or uninstall without prompting\n  \
           --no-launch     do not offer to start the app after installing\n  \
           --install-dir   install somewhere other than %LOCALAPPDATA%\\Programs\\ArgusNextPublic\n  \
           --verify        re-check every embedded file and exit\n\n\
         The setup program is unsigned and installs for the current user only."
    );
}

#[cfg(test)]
mod tests {
    use super::uninstall_command;
    use std::path::Path;

    #[test]
    fn registered_uninstall_preserves_custom_location() {
        let directory = Path::new(r"C:\Custom Apps\Argus");
        let executable = r"C:\Custom Apps\Argus\argus-uninstall.exe";
        assert_eq!(
            uninstall_command(executable, directory, false),
            r#""C:\Custom Apps\Argus\argus-uninstall.exe" --uninstall --install-dir "C:\Custom Apps\Argus""#
        );
        assert!(uninstall_command(executable, directory, true).ends_with(" --silent"));
        assert!(uninstall_command(executable, Path::new("C:\\"), false)
            .ends_with("--install-dir \"C:\\\\\""));
    }

    #[cfg(windows)]
    #[test]
    fn generated_maintenance_scripts_parse_without_execution() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let directory = Path::new(r"C:\Custom Apps\Kid's Argus");
        for script in [
            super::shared_installation_script().to_owned(),
            super::shortcut_cleanup_script(directory),
            super::registration_cleanup_script(directory),
        ] {
            // ParseInput is syntax-only: do not invoke/evaluate these scripts,
            // inspect account registration, create COM objects, or delete links.
            let mut child = Command::new("powershell.exe")
                .args(["-NoProfile", "-NonInteractive", "-Command",
                    "$source=[Console]::In.ReadToEnd(); $tokens=$null; $errors=$null; [Management.Automation.Language.Parser]::ParseInput($source,[ref]$tokens,[ref]$errors) | Out-Null; if ($errors.Count) { $errors | Out-String | Write-Error; exit 1 }"])
                .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
                .spawn().unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(script.as_bytes())
                .unwrap();
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
