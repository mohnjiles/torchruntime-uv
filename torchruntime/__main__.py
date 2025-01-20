from .installer import install


def print_usage(entry_command: str):
    """Print usage information with examples."""
    usage = """
Usage: {entry_command} <command> [arguments]

Commands:
    install             Install PyTorch packages
    --help             Show this help message

Examples:
    {entry_command} install
    {entry_command} install torch==2.2.0 torchvision==0.17.0
    {entry_command} install torch>=2.0.0 torchaudio
    {entry_command} install torch==2.1.* torchvision>=0.16.0 torchaudio==2.1.0

If no packages are specified, the latest available versions
of torch, torchaudio and torchvision will be installed.

Version specification formats (follows pip format):
    package==2.1.0     Exact version
    package>=2.0.0     Minimum version
    package<=2.2.0     Maximum version
    package~=2.1.0     Compatible release
    package==2.1.*     Any 2.1.x version
    package            Latest version
    """.format(entry_command=entry_command)
    print(usage.strip())


def main():
    import sys

    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
        # get the entry point
        entry_path = sys.argv[0]
        if "__main__.py" in entry_path:
            cli = "python -m torchruntime"
        else:
            cli = "torchruntime"
        print_usage(cli)
        return

    command = sys.argv[1]

    if command == "install":
        # Pass all arguments after 'install' to the install function
        package_versions = sys.argv[2:] if len(sys.argv) > 2 else None
        install(package_versions)
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
