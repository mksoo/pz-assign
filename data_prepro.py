# modified by mksoo
# https://github.com/mksoo

# Copyright 2022 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
midi --> tfrecord
사용예시:
  $ python data_prepro.py --input_dir <input_dir> --recursive
"""

import hashlib
import os

import pandas as pd
from note_seq import midi_io
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('input_dir', None,
                    'Directory containing files to convert.')
flags.DEFINE_string('output_file', None,
                    'Path to output TFRecord file. Will be overwritten '
                    'if it already exists.')
flags.DEFINE_bool('recursive', False,
                  'Whether or not to recurse into subdirectories.')

if __name__ == '__main__':
    flags.DEFINE_string(
        'log', 'INFO',
        'The threshold for what messages will be logged: '
        'DEBUG, INFO, WARN, ERROR, or FATAL.')


def generate_note_sequence_id(filename, collection_name, source_type):
    """sequence에 대해서 각각의 ID 부여
    format:'/id/<type>/<collection name>/<hash>'.

    Args:
    filename: The string path to the source file relative to the root of the
        collection.
    collection_name: The collection from which the file comes.
    source_type: The source type as a string (e.g. "midi" or "abc").

    Returns:
    The generated sequence ID as a string.
    """
    filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
    return '/id/%s/%s/%s' % (
        source_type.lower(), collection_name, filename_fingerprint.hexdigest())


def convert_midi(root_dir, sub_dir, full_file_path):
    """Converts a midi file to a sequence proto.

    Args:
    root_dir: A string specifying the root directory for the files being
        converted.
    sub_dir: The directory being converted currently.
    full_file_path: the full path to the file to convert.

    Returns:
    Either a NoteSequence proto or None if the file could not be converted.
    """
    try:
        sequence = midi_io.midi_to_sequence_proto(
            tf.gfile.GFile(full_file_path, 'rb').read())
    except midi_io.MIDIConversionError as e:
        tf.logging.warning(
            'Could not parse MIDI file %s. It will be skipped. Error was: %s',
            full_file_path, e)
        return None
    sequence.collection_name = os.path.basename(root_dir)
    sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
    sequence.id = generate_note_sequence_id(
        sequence.filename, sequence.collection_name, 'midi')
    tf.logging.info('Converted MIDI file %s.', full_file_path)
    return sequence


def convert_files(root_dir, sub_dir, writer, f_list, recursive=False):
    """Converts files.
  mksoo: 여기서 train/val/test이면 진행, 아니면 스킵
  Args:
    root_dir: A string specifying a root directory.
    sub_dir: A string specifying a path to a directory under `root_dir` in which
        to convert contents.
    writer: A TFRecord writer
    f_list: 바꿀거 리스트
    recursive: A boolean specifying whether or not recursively convert files
        contained in subdirectories of the specified directory.

  Returns:
    A map from the resulting Futures to the file paths being converted.
  """

    dir_to_convert = os.path.join(root_dir, sub_dir)
    tf.logging.info("Converting files in '%s'.", dir_to_convert)
    files_in_dir = tf.gfile.ListDirectory(os.path.join(dir_to_convert))
    recurse_sub_dirs = []
    written_count = 0
    # 디렉토리 안에 있는 모든 파일들에 대해서 convert를 진행한다.
    for file_in_dir in files_in_dir:
        tf.logging.log_every_n(tf.logging.INFO, '%d files converted.',
                               1000, written_count)
        full_file_path = os.path.join(dir_to_convert, file_in_dir)  # file_path
        # mid || midi 파일일 때
        if (full_file_path.lower().endswith('.mid') or
                full_file_path.lower().endswith('.midi')):
            relative_file_path = os.path.join(sub_dir, file_in_dir)

            # 해당되는 file_list에 없으면
            if relative_file_path.lower() not in f_list:
                tf.logging.warning("file %s is not in f_list.", full_file_path)
                continue
            try:
                sequence = convert_midi(root_dir, sub_dir, full_file_path)
            except Exception as exc:  # pylint: disable=broad-except
                tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
                continue
            # 여기서 저장
            if sequence:
                writer.write(sequence.SerializeToString())
        # 그외: dir || convert 불가.
        else:
            if recursive and tf.gfile.IsDirectory(full_file_path):
                recurse_sub_dirs.append(os.path.join(sub_dir, file_in_dir))
            else:
                tf.logging.warning(
                    'Unable to find a converter for file %s', full_file_path)

    for recurse_sub_dir in recurse_sub_dirs:
        convert_files(root_dir, recurse_sub_dir, writer, f_list, recursive)


def convert_directory(root_dir, recursive=False):
    """Converts files to NoteSequences and writes to `output_file`.

    Input files found in `root_dir` are converted to NoteSequence protos with the
    basename of `root_dir` as the collection_name, and the relative path to the
    file from `root_dir` as the filename. If `recursive` is true, recursively
    converts any subdirectories of the specified directory.

    Args:
    root_dir: A string specifying a root directory.
    output_file: Path to TFRecord file to write results to.
    recursive: A boolean specifying whether or not recursively convert files
        contained in subdirectories of the specified directory.
    """

    df = pd.read_csv("data\groove\info.csv", encoding='cp949')

    midi_filenames = df.midi_filename.tolist()
    dataset_types = df.split.tolist()

    train_list = []
    test_list = []
    val_list = []

    for i in range(len(midi_filenames)):
        if dataset_types[i] == 'train':
            train_list.append(midi_filenames[i])
        elif dataset_types[i] == 'test':
            test_list.append(midi_filenames[i])
        else:
            val_list.append(midi_filenames[i])

    f_list = [train_list,
              test_list,
              val_list
              ]
    dataset_names = ['data/train.tfrecord',
                     'data/test.tfrecord',
                     'data/val.tfrecord'
                     ]

    for i in range(len(f_list)):
        output_file = dataset_names[i]
        with tf.io.TFRecordWriter(output_file) as writer:
            convert_files(root_dir, '', writer, f_list[i], recursive)


def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)

    if not FLAGS.input_dir:
        tf.logging.fatal('--input_dir required')
        return

    input_dir = os.path.expanduser(FLAGS.input_dir)

    convert_directory(input_dir, FLAGS.recursive)


def console_entry_point():
    tf.disable_v2_behavior()
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
