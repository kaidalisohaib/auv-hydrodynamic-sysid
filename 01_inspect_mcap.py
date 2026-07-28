#!/usr/bin/env python3
"""
01_inspect_mcap.py - MCAP ROS 2 Bag Inspector
McGill Robotics | douglas-sysid pipeline

Seeks to the middle of a ROS 2 bag file and prints one sample message
from every valid topic. Useful for verifying bag contents before
running the extraction pipeline.

Usage:
    python 01_inspect_mcap.py path/to/bag
"""

import os
import argparse
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


# Topics to skip during inspection (noisy or irrelevant)
SKIP_PREFIXES = ('/vision/', 'foxglove')
SKIP_EXACT = {'/planner_root_tree/snapshots'}


def get_middle_messages(bag_path):
    """Seek to the temporal midpoint of a bag and print one message per topic."""
    if not os.path.exists(bag_path):
        print(f"Error: Bag path '{bag_path}' does not exist.")
        return

    # 1. Open the bag reader
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    # 2. Compute the middle timestamp from metadata
    metadata = reader.get_metadata()
    start_nanos = metadata.starting_time.nanoseconds
    duration_nanos = metadata.duration.nanoseconds
    middle_timestamp = start_nanos + (duration_nanos // 2)

    # 3. Build topic → type mapping
    topic_types = {
        topic.topic_metadata.name: topic.topic_metadata.type
        for topic in metadata.topics_with_message_count
    }

    # 4. Determine which topics to inspect
    target_topics = {
        t for t in topic_types.keys()
        if not any(t.startswith(p) for p in SKIP_PREFIXES) and t not in SKIP_EXACT
    }

    reader.seek(middle_timestamp)
    printed_topics = set()

    print(f"Bag duration: {duration_nanos / 1e9:.1f}s | Seeking to middle: {middle_timestamp}")
    print(f"Inspecting {len(target_topics)} topics...\n" + "=" * 50)

    # 5. Read forward from the midpoint, capturing one message per topic
    while reader.has_next() and len(printed_topics) < len(target_topics):
        topic, data, timestamp = reader.read_next()

        if any(topic.startswith(p) for p in SKIP_PREFIXES) or topic in SKIP_EXACT:
            continue

        if topic not in printed_topics:
            printed_topics.add(topic)
            msg_type_str = topic_types[topic]
            try:
                msg_type = get_message(msg_type_str)
                msg = deserialize_message(data, msg_type)

                print(f"\n[TOPIC]: {topic} ({msg_type_str})")
                print(f"[TIMESTAMP]: {timestamp}")
                print(msg)
                print("-" * 40)
            except Exception as e:
                print(f"\nCould not parse topic {topic}: {e}")

    print(f"\nFinished. Printed messages for {len(printed_topics)} / {len(target_topics)} topics.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspect the middle message of every topic in an MCAP ROS 2 bag.")
    parser.add_argument("bag_path", type=str,
                        help="Path to the ROS 2 bag directory or .mcap file.")
    args = parser.parse_args()

    get_middle_messages(args.bag_path)
