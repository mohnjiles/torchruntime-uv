# Converts the txt file at https://github.com/pciutils/pciids/blob/master/pci.ids to an sqlite3 db

import sqlite3
import re
import sys

AMD = "1002"
NVIDIA = "10de"
INTEL = "8086"

# Compile all regular expressions
ARC_REGEX = re.compile(r"\barc(\b|\W)")
SKIP_LINE_REGEX = re.compile(r"^\t{2,}")
VENDOR_REGEX = re.compile(r"^([0-9a-fA-F]{4})\s+(.*)")
DEVICE_REGEX = re.compile(r"^\t([0-9a-fA-F]{4})\s+(.*)")
GPU_NAME_REGEXES = {
    AMD: {
        "include": re.compile(r"\b(?:radeon|instinct|fire|rage|polaris)\b", re.IGNORECASE),
        "exclude": re.compile(r"\b(?:audio|bridge)\b", re.IGNORECASE),
    },
    INTEL: {
        "include": re.compile(r"\b(?:arc)\b", re.IGNORECASE),
        "exclude": re.compile(r"\b(?:audio|bridge)\b", re.IGNORECASE),
    },
    NVIDIA: {
        "include": re.compile(r"\b(?:geforce|riva|quadro|tesla|ion|grid|rtx|tu\d{2,}.+t\d{2,})\b", re.IGNORECASE),
        "exclude": re.compile(
            r"\b(?:audio|switch|pci|memory|smbus|ide|co-processor|bridge|usb|sata|controller)\b", re.IGNORECASE
        ),
    },
}


def process_pci_file(input_file: str, output_db: str):
    # Connect to the SQLite database (create if it doesn't exist)
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pci_ids (
            vendor_id TEXT,
            vendor_name TEXT,
            device_id TEXT,
            device_name TEXT
        )
    """
    )

    vendor_id, vendor_name = None, None  # Initialize the current vendor context

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            # Skip comments and deeply indented lines (2 or more indents)
            if line.startswith("#") or SKIP_LINE_REGEX.match(line):
                continue

            # Match vendor lines (no tab)
            match_vendor = VENDOR_REGEX.match(line)
            if match_vendor:
                vendor_id, vendor_name = match_vendor.groups()
                continue  # Move to next line for processing

            if vendor_id not in (NVIDIA, INTEL, AMD):
                continue

            # Match device lines (single tab)
            match_device = DEVICE_REGEX.match(line)
            if match_device and vendor_id and vendor_name:
                device_id, device_name = match_device.groups()

                include_regex = GPU_NAME_REGEXES[vendor_id]["include"]
                exclude_regex = GPU_NAME_REGEXES[vendor_id]["exclude"]

                if not include_regex.search(device_name) or exclude_regex.search(device_name):
                    continue

                # Insert the vendor and device information into the database
                cursor.execute(
                    """
                    INSERT INTO pci_ids (vendor_id, vendor_name, device_id, device_name)
                    VALUES (?, ?, ?, ?)
                """,
                    (vendor_id, vendor_name, device_id, device_name),
                )

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Check if input and output arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_db>")
        print("Get the text file at https://github.com/pciutils/pciids/blob/master/pci.ids")
        sys.exit(1)

    input_file = sys.argv[1]
    output_db = sys.argv[2]

    # Process the input file and output to the SQLite database
    process_pci_file(input_file, output_db)
