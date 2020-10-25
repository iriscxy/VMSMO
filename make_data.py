import sys
import os
import hashlib
import struct
import subprocess
import collections
import json
import tensorflow as tf
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
CHUNK_SIZE = 1000

chunks_dir = 'bin_files'


def chunk_file(set_name):
    in_file = '%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    # Chunk the data
    for set_name in ['multi_train',  'multi_test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)



def write_to_bin(url_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    f=open(url_file)
    lines=f.readlines()

    with open(out_file, 'wb') as writer:
        for line in lines:
            content=json.loads(line)
            abstract=content['summary']
            abstract = "%s %s %s" % (SENTENCE_START, abstract, SENTENCE_END)
            article=content['content']
            multi1=content['multi1']
            multi2=content['multi2']


            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example.features.feature['multi1'].bytes_list.value.extend([multi1.encode()])
            tf_example.features.feature['multi2'].bytes_list.value.extend([multi2.encode()])

            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))


    print("Finished writing file %s\n" % out_file)


if __name__ == '__main__':


    # Read the tokenized stories, do a little postprocessing then write to bin files
    write_to_bin('/home1/chenxy/project/cnndm/train_multi.json', 'multi_train.bin')
    write_to_bin('/home1/chenxy/project/cnndm/select.json','multi_test.bin')

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()
