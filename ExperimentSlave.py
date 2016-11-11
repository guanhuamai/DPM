import socket
import threading
from struct import unpack
from struct import pack

def pose_task(id_exp):
    id_w = id_exp / 10
    id_b = (id_exp % 10) / 2
    id_d = id_exp % 2
    return str((id_w, id_b, id_d)) + '\n'


def fuck():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = ('127.0.0.1', 9999)
    s.connect(addr)

    while True:
        id_bytes = s.recv(1024)

        if id_bytes is None or len(id_bytes) == 0 or (len(id_bytes) == 4 and unpack('i', id_bytes)[0] == -1):
            print 'no more experiments received, slaver life ends...'
            break

        id_exp = unpack('i', id_bytes)[0]

        rslt = pose_task(id_exp)

        s.send(rslt.encode('utf-8'))

    s.send(pack('i', -1))
    s.close()


if __name__ == '__main__':
    for i in range(5):
        t = threading.Thread(target=fuck)
        t.start()


