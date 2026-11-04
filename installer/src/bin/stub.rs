//! Per-user Windows setup program for the Argus Next public prototype.
//!
//! The payload is appended by `argus-setup-builder`. Every file is verified
//! against its recorded SHA-256 before anything is written, so a truncated or
//! altered download fails instead of installing partially.

use argus_setup::container::{sanitize_relative, Package};
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
                options.install_dir = Some(PathBuf::from(value));
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

    println!("{APP_NAME} {version}");
    println!("Install location: {}", directory.display());
    if !options.silent {
        println!("This installs for the current user and needs no administrator rights.");
        if !confirm("Continue? [Y/n] ", true) {
            println!("Cancelled.");
            return Ok(());
        }
    }

    fs::create_dir_all(&directory)?;
    package.extract(|relative| Ok(directory.join(sanitize_relative(relative)?)))?;
    write_install_record(&directory, version, package)?;
    create_shortcuts(&directory)?;
    // Copy only the stub, not the 21 MB payload, so the installed uninstaller
    // stays small while still working after the download is deleted.
    let uninstaller = directory.join("argus-uninstall.exe");
    copy_prefix(setup, &uninstaller, package.stub_length())?;
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

    let remove_data =
        !options.silent && confirm("Also delete captured data and settings? [y/N] ", false);
    remove_shortcuts()?;
    remove_uninstall_entry();
    remove_directory(&directory)?;

    let data = default_data_directory()?;
    if remove_data && data.exists() {
        fs::remove_dir_all(&data)?;
        println!("Removed data: {}", data.display());
    } else {
        println!("Kept data: {}", data.display());
    }
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

fn remove_shortcuts() -> io::Result<()> {
    if let Some(appdata) = env::var_os("APPDATA") {
        let link = PathBuf::from(appdata)
            .join(r"Microsoft\Windows\Start Menu\Programs")
            .join("Argus Next Public.lnk");
        if link.exists() {
            fs::remove_file(link)?;
        }
    }
    let script = "$ErrorActionPreference='SilentlyContinue'; \
                  $desktop=[Environment]::GetFolderPath('Desktop'); \
                  if ($desktop) { \
                    $link=Join-Path $desktop 'Argus Next Public.lnk'; \
                    if (Test-Path -LiteralPath $link) { Remove-Item -LiteralPath $link -Force } \
                  }";
    let _ = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .output();
    Ok(())
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
            format!("\"{setup_path}\" --uninstall"),
            "REG_SZ",
        ),
        (
            "QuietUninstallString",
            format!("\"{setup_path}\" --uninstall --silent"),
            "REG_SZ",
        ),
        ("EstimatedSize", size_kb.to_string(), "REG_DWORD"),
    ];
    for (name, value, kind) in values {
        reg(&["add", UNINSTALL_KEY, "/f", "/v", name, "/t", kind, "/d", &value])?;
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

fn remove_uninstall_entry() {
    let _ = reg(&["delete", UNINSTALL_KEY, "/f"]);
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

fn remove_directory(directory: &Path) -> io::Result<()> {
    if !directory.exists() {
        return Ok(());
    }
    let mut last_error = None;
    for attempt in 0..20 {
        match fs::remove_dir_all(directory) {
            Ok(()) => return Ok(()),
            Err(error) => {
                if attempt < 19 {
                    last_error = Some(error);
                    sleep(Duration::from_millis(250));
                } else {
                    return Err(error);
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| io::Error::other("could not remove the install directory")))
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
    Ok(PathBuf::from(local).join("Programs").join("ArgusNextPublic"))
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
