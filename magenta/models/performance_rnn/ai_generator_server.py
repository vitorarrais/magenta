import socket
import socketserver
import os
import logging
import json
from magenta.models.performance_rnn import ai_generator as generator

# prepare simple logging
logger = logging.getLogger('SOCKET_SERVER')
handler = logging.StreamHandler()
formatter = logging.Formatter( \
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

speed = 5

class RequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        global speed
        self.data = self.request.recv(1024).strip()
        logger.debug('Received chunk of data: %s' % self.data)
        recv_chunk = json.loads(self.data.decode('utf-8'))
        if 'speed' in recv_chunk :
            # avoid slowing down the speed to zero 
            if int(recv_chunk['speed']) > 0:
                speed = recv_chunk['speed']
        else:
            # create an in-memory song file
            logger.debug('Using speed: %s' % speed)
            generated_song = generator.generate(speed)
            # find out the file's length
            generated_song.seek(0, os.SEEK_END)
            buffer_size = generated_song.tell()
            generated_song.seek(0)
            logger.debug('Request buffer of size: %s' % buffer_size)
            # inform the buffersize to the client
            buff_data = {'bufferSize' : buffer_size}
            self.request.send(bytearray(json.dumps(buff_data), 'utf-8'))
            # then send the entire song
            result = self.request.sendall(generated_song.read())
            if result is None:
                logger.debug('Chunk sent successfully.')
            generated_song.close()


if __name__ == '__main__':
    HOST, PORT = 'localhost', 9999

    server = socketserver.ThreadingTCPServer((HOST,PORT), RequestHandler)
    server.socket_type = socket.SOCK_STREAM
    logger.debug('Listening to %s:%s' %(HOST,PORT))
    server.serve_forever()
