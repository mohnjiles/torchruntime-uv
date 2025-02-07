import os
import re
import json
import sqlite3
import platform
import subprocess

from .consts import NVIDIA, AMD, INTEL

DEVICE_DB_FILE = "gpu_pci_ids.db"  # this file will only include AMD, NVIDIA and Discrete Intel GPUs

GPU_DEVICES = {  # if the value is a regex, it'll be applied to the device_name. if the value is a dict, the pci_id will be looked up
    AMD: {
        "discrete": re.compile(r"\b(?:radeon|instinct|fire|rage|polaris|aldebaran)\b", re.IGNORECASE),
        "integrated": {  # pci_id -> (device_name, gfx_name, hsa_override)
            "9874": ("Wani", "gfx801", "8.0.1"),
            "98e4": ("Stoney", "gfx810", "8.1.0"),
            "15dd": ("Raven Ridge", "gfx902", "9.1.0"),
            "15d8": ("Picasso", "gfx903", "9.1.0"),
            "1636": ("Renoir", "gfx90c", "9.3.0"),
            "164c": ("Lucienne", "gfx90c", "9.3.0"),
            "1638": ("Cezanne", "gfx90c", "9.3.0"),
            "15e7": ("Barcelo", "gfx90c", "9.3.0"),
            "13f9": ("Oberon", "gfx1013", "10.1.3"),
            "1607": ("Arden", "gfx1020", "10.2.0"),
            "163f": ("VanGogh", "gfx1033", "10.3.1"),
            "164d": ("Rembrandt", "gfx1035", "10.3.3"),
            "1681": ("Rembrandt", "gfx1035", "10.3.3"),
            "164e": ("Raphael", "gfx1036", "10.3.6"),
            "1506": ("Mendocino", "gfx1037", "10.3.7"),
            "164f": ("Phoenix", "gfx1103", "11.0.1"),
            "15bf": ("Phoenix1", "gfx1103", "11.0.1"),
            "15c8": ("Phoenix2", "gfx1103", "11.0.4"),
        },
        "exclude": re.compile(r"\b(?:audio|bridge)\b", re.IGNORECASE),
    },
    INTEL: {
        "discrete": re.compile(r"\b(?:arc)\b", re.IGNORECASE),
        "integrated": re.compile(r"\b(?:iris|hd graphics|uhd graphics)\b", re.IGNORECASE),
        "exclude": re.compile(r"\b(?:audio|bridge)\b", re.IGNORECASE),
    },
    NVIDIA: {
        "discrete": re.compile(r"\b(?:geforce|riva|quadro|tesla|ion|grid|rtx|tu\d{2,}.+t\d{2,})\b", re.IGNORECASE),
        "exclude": re.compile(
            r"\b(?:audio|switch|pci|memory|smbus|ide|co-processor|bridge|usb|sata|controller)\b", re.IGNORECASE
        ),
    },
}


os_name = platform.system()


def is_gpu_vendor(vendor_id):
    return vendor_id in GPU_DEVICES


def get_gpu_type(vendor_id, device_id, device_name):
    """
    Returns the GPU type of the given PCI device.

    Args:
        vendor_id (str): PCI Vendor ID.
        device_id (str): PCI Device ID.
        device_name (str): PCI Device Name.

    Returns:
        gpu_type (str): "DISCRETE", "INTEGRATED", or "NONE"
    """

    def matches(pattern):
        if isinstance(pattern, re.Pattern):
            return pattern.search(device_name)
        if isinstance(pattern, dict):
            return device_id in pattern
        return False

    vendor_devices = GPU_DEVICES[vendor_id]

    discrete_devices = vendor_devices.get("discrete")
    integrated_devices = vendor_devices.get("integrated")
    exclude_devices = vendor_devices.get("exclude")

    if matches(exclude_devices):
        return "NONE"

    if matches(integrated_devices):  # check integrated first, to avoid matching "Radeon" and classifying it as discrete
        return "INTEGRATED"

    if matches(discrete_devices):
        return "DISCRETE"

    return "NONE"


def get_windows_output():
    try:
        command = ["powershell", "-Command", "Get-WmiObject Win32_VideoController | ForEach-Object { $_.PNPDeviceID }"]
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return ""


def get_linux_output():
    try:
        return subprocess.check_output(["lspci", "-nn"], text=True)
    except FileNotFoundError:
        return ""


def get_macos_output():
    try:
        return subprocess.check_output(["system_profiler", "-json", "SPDisplaysDataType"], text=True)
    except subprocess.CalledProcessError:
        return ""


def parse_windows_output(output):
    pci_ids = set()
    for line in output.splitlines():
        match = re.search(r"VEN_(\w+)&DEV_(\w+)", line, re.IGNORECASE)
        if match:
            vendor_id = match.group(1).lower()
            device_id = match.group(2).lower()
            pci_ids.add((vendor_id, device_id))
    return list(pci_ids)


def parse_linux_output(output):
    pci_ids = set()
    for line in output.splitlines():
        match = re.search(r"\[(\w+):(\w+)\]", line)
        if match:
            vendor_id = match.group(1).lower()
            device_id = match.group(2).lower()
            pci_ids.add((vendor_id, device_id))
    return list(pci_ids)


def parse_macos_output(output):
    pci_ids = set()
    try:
        data = json.loads(output)
        displays = data.get("SPDisplaysDataType", [])
        for display in displays:
            vendor_raw = display.get("spdisplays_vendor", "")
            device_id_raw = display.get("spdisplays_device-id", "")
            if device_id_raw and vendor_raw:
                device_id = device_id_raw.replace("0x", "").lower()
                if "Intel" in vendor_raw:
                    vendor_id = "8086"
                else:
                    match = re.search(r"\((0x\w+)\)", vendor_raw)
                    if match:
                        vendor_id = match.group(1).replace("0x", "").lower()
                    else:
                        continue
                pci_ids.add((vendor_id, device_id))
    except json.JSONDecodeError:
        pass
    return list(pci_ids)


def get_pci_ids():
    if os_name == "Windows":
        output = get_windows_output()
        return parse_windows_output(output)
    elif os_name == "Linux":
        output = get_linux_output()
        return parse_linux_output(output)
    elif os_name == "Darwin":  # macOS
        output = get_macos_output()
        return parse_macos_output(output)
    else:
        return []


def get_device_infos(pci_ids):
    """
    Reads the given SQLite database file and queries the `pci_ids` table
    for matching vendor_id and device_id. Returns a list of tuples containing
    (vendor_id, vendor_name, device_id, device_name).

    Args:
        db_file_name (str): Path to the SQLite database file.
        pci_ids (list of tuples): List of (vendor_id, device_id) pairs to match.

    Returns:
        list of tuples: List of (vendor_id, vendor_name, device_id, device_name).
    """
    result = []

    # Establish connection to the database
    db_path = os.path.join(os.path.dirname(__file__), DEVICE_DB_FILE)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create a query to retrieve matching rows
        query = """
        SELECT vendor_id, vendor_name, device_id, device_name
        FROM pci_ids
        WHERE vendor_id = ? AND device_id = ?
        """

        # Execute query for each (vendor_id, device_id) in pci_ids
        for vendor_id, device_id in pci_ids:
            cursor.execute(query, (vendor_id, device_id))
            rows = cursor.fetchall()
            result.extend(rows)

    finally:
        # Close the database connection
        conn.close()

    return result


def get_discrete_gpus():
    pci_ids = get_pci_ids()
    return get_device_infos(pci_ids)
