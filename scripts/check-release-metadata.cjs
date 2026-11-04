#!/usr/bin/env node
// Asserts that every place a version, toolchain, or license is declared agrees.
// Run locally with `node scripts/check-release-metadata.cjs`; CI runs it too.

"use strict";

const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const problems = [];

const read = (relative) => fs.readFileSync(path.join(root, relative), "utf8");
const capture = (source, pattern) => {
  const match = pattern.exec(source);
  return match ? match[1] : null;
};

const metadata = JSON.parse(
  execFileSync("cargo", ["metadata", "--no-deps", "--format-version", "1"], {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  }),
);

const versions = new Set(metadata.packages.map((pkg) => pkg.version));
if (versions.size !== 1) {
  problems.push(`workspace crate versions disagree: ${[...versions].join(", ")}`);
}
const crateVersion = [...versions][0];

const licenses = new Set(metadata.packages.map((pkg) => pkg.license));
if (licenses.size !== 1 || !licenses.has("Apache-2.0")) {
  problems.push(`workspace crate licenses disagree: ${[...licenses].join(", ")}`);
}

const tauriVersion = JSON.parse(read("desktop/src-tauri/tauri.conf.json")).version;
if (tauriVersion !== crateVersion) {
  problems.push(`desktop/src-tauri/tauri.conf.json is ${tauriVersion}, crates are ${crateVersion}`);
}

const installerVersion = capture(read("installer/Cargo.toml"), /^version\s*=\s*"([^"]+)"/m);
if (installerVersion !== crateVersion) {
  problems.push(`installer/Cargo.toml is ${installerVersion}, crates are ${crateVersion}`);
}

const rustVersion = capture(read("Cargo.toml"), /^rust-version\s*=\s*"([^"]+)"/m);
const channel = capture(read("rust-toolchain.toml"), /^channel\s*=\s*"([^"]+)"/m);
if (!channel) {
  problems.push("rust-toolchain.toml has no channel");
} else if (rustVersion && !channel.startsWith(`${rustVersion}.`)) {
  problems.push(`rust-toolchain.toml pins ${channel}, Cargo.toml requires ${rustVersion}`);
}

const changelog = read("CHANGELOG.md");
if (!changelog.includes(crateVersion)) {
  problems.push(`CHANGELOG.md mentions no release containing ${crateVersion}`);
}

for (const required of ["LICENSE", "NOTICE", "SECURITY.md", "docs/ROADMAP.md"]) {
  if (!fs.existsSync(path.join(root, required))) {
    problems.push(`missing ${required}`);
  }
}

if (!/Apache License/.test(read("LICENSE"))) {
  problems.push("LICENSE does not contain the Apache License text");
}

if (problems.length > 0) {
  for (const problem of problems) console.error(`metadata check failed: ${problem}`);
  process.exit(1);
}

console.log(
  `Release metadata consistent: version ${crateVersion}, Rust ${channel}, license Apache-2.0.`,
);
