# Argus Next

Argus Next is a Windows app I'm building to make network activity easier to inspect. It brings packet capture, flow information and alerts into one place. I'm especially interested in whether a tool like this could be useful to smaller organizations that don't have a dedicated security team.

## What it does

The app can capture traffic, show network flows and replay packet captures. It also includes models that forecast packet behavior. One of them, Adaptive Hierarchical Event Memory (AHEM), is the focus of my research into how a small model can use a flow's earlier history.

A forecast isn't the same thing as detecting an attack. Argus is still a prototype: I haven't yet established how well it detects threats in an organization's network or how often it produces false alerts. Please don't rely on it as your only security tool.

## Privacy and pilots

I'm interested in working with organizations that would like to explore a pilot. Before any capture, we'd agree on what traffic can be monitored, who can access the data, how long it's kept and how findings are handled. Network metadata can be sensitive; only capture traffic you have permission to monitor.

## Availability

Download the setup program from the [latest release](https://github.com/blah-blah-cell/argus-next-public/releases) and run it. It installs for the current user without administrator rights, adds Start menu and desktop shortcuts, and starts with capture stopped. See the [Windows install guide](docs/INSTALL_WINDOWS.md).

The setup program is unsigned, so Windows may show a SmartScreen warning. The source and two AHEM forecasting models are in this repository; to build from source, follow the [Windows setup guide](docs/BUILD_WINDOWS.md). Model forecasting needs the optional Python runtime described there, and the app reports model readiness rather than hiding it.

The public build uses its own settings and data directories. Monitoring starts only when you choose to start it. The newer AHEM model is experimental and disabled by default; it doesn't make security decisions.

## Research

The [research overview](research/README.md) explains what AHEM has been tested on and what those tests can—and can't—tell us.

## Roadmap

The public plan is in [docs/ROADMAP.md](docs/ROADMAP.md). It covers signing and supply-chain trust, removing the Python runtime from the installed app, retention and alert delivery, a multi-site sensor and collector, and Linux and macOS work.

Windows is the current target. Linux and macOS support is in progress and those platforms aren't ready for pilots yet.
