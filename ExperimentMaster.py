import threading
from struct import pack
from struct import unpack
import socket

id_lock = threading.Lock()
id_exp = 0  # maximum id_exp = 29, 30 experiments in total

rslts_lock = threading.Lock()
rslts = ''


def read_id_exp():
    global id_exp, id_lock
    id_lock.acquire()
    rslt = id_exp
    id_exp += 1
    print id_exp
    id_lock.release()
    return rslt


def alloc_task(sock_conn, addr):
    global id_exp, id_lock, rslts, rslts_lock
    while True:
        cur_id = read_id_exp()
        if cur_id >= 30:
            break

        print 'allocate %d to %s:%s\n' % (cur_id, addr[0], addr[1])

        sock_conn.send(pack('i', cur_id))

        rslt = sock_conn.recv(1024)

        if not rslt or (len(rslt) == 4 and unpack('i', rslt)[0] == -1):
            print ' %d to %s:%s, failed!\n' % (cur_id, addr[0], addr[1])
            break
        else:
            rslt = rslt.decode('utf-8')
            rslts_lock.acquire()
            rslts += rslt
            rslts_lock.release()
    sock_conn.send(pack('i', -1))
    print 'end %s:%s\n' % (addr[0], addr[1])
    sock_conn.close()


if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 9999))
    sock.listen(100)
    #while True:
    _sock_conn, _addr = sock.accept()
    print 'things...'
    t = threading.Thread(target=alloc_task, args=(_sock_conn, _addr))
    t.start()


    print rslts
