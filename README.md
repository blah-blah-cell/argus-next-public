# Argus Next

Argus Next is a Windows app I'm building to make network activity easier to inspect. It brings packet capture, flow information and alerts into one place. I'm especially interested in whether a tool like this could be useful to smaller organizations that don't have a dedicated security team.

## What it does

The app can capture traffic, show network flows and replay packet captures. It also includes models that forecast packet behavior. One of them, Adaptive Hierarchical Event Memory (AHEM), is the focus of my research into how a small model can use a flow's earlier history.

A forecast isn't the same thing as detecting an attack. Argus is still a prototype: I haven't yet established how well it detects threats in an organization's network or how often it produces false alerts. Please don't rely on it as your only security tool.

## Privacy and pilots

I'm interested in working with organizations that would like to explore a pilot. Before any capture, we'd agree on what traffic can be monitored, who can access the data, how long it's kept and how findings are handled. Network metadata can be sensitive; only capture traffic you have permission to monitor.

## Availability

The source and two AHEM forecasting models are in this repository. To build and run the Windows prototype, follow the [Windows setup guide](docs/BUILD_WINDOWS.md). There isn't a one-click installer yet.

The public build uses its own settings and data directories. Monitoring starts only when you choose to start it. The newer AHEM model is experimental and disabled by default; it doesn't make security decisions.

## Research

The [research overview](research/README.md) explains what AHEM has been tested on and what those tests can—and can't—tell us.

## Roadmap

- Make the Windows setup and first run easier on a fresh computer.
- Work with pilot organizations to understand which views and alerts are useful.
- Test detection quality, false alerts and resource use on independent traffic.
- Add Linux and macOS support (in progress). Windows is the current target; the other platforms aren't ready for pilots yet.
