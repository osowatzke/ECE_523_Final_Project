import subprocess
import os
import re

def read_table(tbl_str, col_names):
    tbl = {key: [] for key in col_names}
    idx = tbl.copy()
    col_idx = None
    sep_found = False
    cmd_successful = False
    sep = '\\s+'
    regex_str = ['(' + col_name + ')' for col_name in col_names]
    regex_str = '^\\s*' + sep.join(regex_str) + '\\s*$'

    for line in tbl_str.splitlines():
        if col_idx is None:
            m = re.match(regex_str, line)
            if m is not None:
                s = 0
                col_idx = []
                for idx in range(len(col_names)):
                    col_idx.append(line.index(m.group(idx+1), s))
                    s = s + len(m.groups(idx+1))
        elif not sep_found:
            s = col_idx[0]
            e = col_idx[-1]
            sep = "-" * (e - s)
            if line[s:e] == sep:
                sep_found = True
        elif line == "The command completed successfully.":
            cmd_successful = True
            break
        else:
            for idx in range(len(col_idx)):
                s = col_idx[idx]
                if idx == (len(col_idx) - 1):
                    e = -1
                else:
                    e = col_idx[idx+1]
                col_name = col_names[idx]
                tbl[col_name].append(line[s:e].strip())
    if (col_idx is not None) and sep_found and cmd_successful:
        return tbl
    else:
        return None

def get_net_shares():
    s = subprocess.check_output(['net', 'share']).decode()
    t = read_table(s, ['Share name', 'Resource', 'Remark'])
    net_shares = {}
    if t is not None:
        for idx in range(len(t['Share name'])):
            net_shares[t['Share name'][idx]] = t['Resource'][idx]
    return net_shares

def get_mapped_drives():
    s = subprocess.check_output(['net', 'use']).decode()
    t = read_table(s, ['Status', 'Local', 'Remote','Network'])
    mapped_drives = {}
    if t is not None:
        for idx in range(len(t['Status'])):
            mapped_drives[t['Local'][idx]] = t['Remote'][idx]
    return mapped_drives

def split_path(path):
    path = os.path.abspath(path)
    parts = path.split(os.sep)
    empty_count = 0
    for part in parts:
        if part == '':
            empty_count += 1
        else:
            break
    parts = parts[empty_count::]
    parts[0] = (os.sep * empty_count) + parts[0]
    empty_count = 0
    for idx in range(len(parts)-1,-1,-1):
        part = parts[idx]
        if part == '':
            empty_count += 1
        else:
            break
    e = len(parts) - empty_count
    parts = parts[:e]
    return parts

def replace_mapped_drives(path):
    parts = split_path(path)
    mapped_drives = get_mapped_drives()
    for key, val in zip(mapped_drives.keys(), mapped_drives.values()):
        if parts[0] == key:
            parts[0] = val
            break
    path = os.sep.join(parts)
    parts = split_path(path)
    net_shares = get_net_shares()
    for idx in range(len(parts)):
        for key, val in zip(net_shares.keys(), net_shares.values()):
            if parts[idx].upper() == key.upper():
                parts[idx] = val
    path = os.sep.join(parts)
    return os.path.abspath(path)

def is_path_remote(path):
    path = replace_mapped_drives(path)
    network_path_start = os.path.sep * 2
    return path.startswith(network_path_start)

if __name__ == "__main__":
    # read_table('col1 col2',['col1','col2'])
    mapped_drives = get_mapped_drives()
    net_shares = get_net_shares()
    print(os.path.abspath('\\apfs\\Users/osowa\\'))
    print(replace_mapped_drives('Z:\\Users/osowa\\'))
    print(is_path_remote('C:/'))
    print(is_path_remote('Z:/'))
    #print(mapped_drives)
    #print(net_shares)