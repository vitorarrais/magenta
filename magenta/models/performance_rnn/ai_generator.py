import socket
import socketserver
import wave
import os
import logging
import json
from magenta.models.performance_rnn import gen as generator

logger = logging.getLogger('SOCKET_SERVER')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class RequestHandler(socketserver.StreamRequestHandler):

    def handle(self):
        self.data = self.request.recv(1024).strip()
        logger.debug('Received chunk of data: %s' % self.data)
        generated_song = generator.generate()
        generated_song.seek(0, os.SEEK_END)
        buffer_size = generated_song.tell()
        generated_song.seek(0)
        logger.debug('Request buffer of size: %s' % buffer_size)
        buff_data = {}
        buff_data['bufferSize'] = buffer_size
        self.request.send(bytearray(json.dumps(buff_data), 'utf-8'))
        result = self.request.sendall(generated_song.read())
        if result is None:
            logger.debug('Chunk sent successfully.')
        generated_song.close()
        # logger.debug(generated_song)
        # if data == 'play':
        # with open('/tmp/performance_rnn/generated/2019-02-01_180505_1.wav', 'rb') as wfile:
        #     buffer_size = os.path.getsize('/tmp/performance_rnn/generated/2019-02-01_180505_1.wav')
        #     logger.debug('Request buffer of size: %s' % buffer_size)
        #     self.request.send(bytes('{ "bufferSize": %s }' % buffer_size, 'utf-8'))
        #     result = self.request.sendall(wfile.read())
        #     if result is None:
        #         logger.debug('Chunk sent successfully.')


if __name__ == '__main__':
    HOST, PORT = 'localhost', 9999

    server = socketserver.TCPServer((HOST,PORT), RequestHandler)
    server.socket_type = socket.SOCK_STREAM
    server.serve_forever()