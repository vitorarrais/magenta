import socket
import socketserver
import wave
import os

class RequestHandler(socketserver.StreamRequestHandler):

    def handle(self):
        self.data = self.request.recv(1024).strip()
        # if data == 'play':
        with open('/tmp/performance_rnn/generated/2019-02-01_180505_1.wav', 'rb') as wfile:
            buffer_size = os.path.getsize('/tmp/performance_rnn/generated/2019-02-01_180505_1.wav')
            print('Total size: %s' % buffer_size)
            self.request.send(bytes('{ "bufferSize": %s }' % buffer_size, 'utf-8'))
            result = self.request.sendall(wfile.read())
            if result is None:
                print('{ "status": "OK" }')


if __name__ == '__main__':
    HOST, PORT = 'localhost', 9999

    server = socketserver.TCPServer((HOST,PORT), RequestHandler)
    server.socket_type = socket.SOCK_STREAM
    server.serve_forever()