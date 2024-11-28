import sys
import requests
from packaging import version

def get_latest_test_version(package_name, version_str):
    url = f"https://test.pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            releases = data["releases"].keys()
            return max((version.parse(v) for v in releases), default=None)
        return None
    except:
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_version.py package_name version")
        sys.exit(1)

    package_name = sys.argv[1]
    version_str = sys.argv[2]

    latest_version = get_latest_test_version(package_name, version_str)
    new_version = version.parse(version_str)

    if latest_version is not None and latest_version >= new_version:
        # If version exists, increment the last number
        components = list(map(int, str(latest_version).split('.')))
        components[-1] += 1
        new_version = '.'.join(map(str, components))
    else:
        new_version = version_str

    print(new_version)

if __name__ == "__main__":
    main()
