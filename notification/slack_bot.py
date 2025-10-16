import os
from pathlib import Path
import json

from swc_slack.messaging.write import send_message_to_user
from swc_slack.utils import get_client
from slack_sdk import WebClient

configs = {
    "slackbot_auth_token_key": "BOT_ID",
    "channel_id": "C09LT3GFS4T"
}

# config
def ensure_default_config(reset_config=False):
    default_config_dir = Path.home() / ".photon_mosaic"
    default_config_path = default_config_dir / "slackbot.json"

    if not default_config_dir.exists():
        raise FileNotFoundError("Run photon-mosaic first to create a config dir!")

    if not default_config_path.exists() or reset_config:
        print(f"✅ Config file written to: {str(default_config_path)}")

        # Write the dictionary to the JSON file
        with default_config_path.open('w', encoding='utf-8') as f:
            json.dump(configs, f)

    return default_config_path


def load_and_process_config():
    # Ensure default config exists
    config_path = ensure_default_config(reset_config=False)

    with config_path.open('r', encoding='utf-8') as f:
        configs = json.load(f)

    return configs

def get_slack_client(configs=None, token=None):
    if token is None:
        client = get_client(configs["slackbot_auth_token_key"])
    else:
        client = get_client(token)

    return client

def test_write(message=None):
    if message is None:
        message = (
            "Hello World!\nHello People!"
            "\n<https://github.com/sumiya-kuroda/photon-mosaic-roicat|GitHub>"
        )

    configs = load_and_process_config()
    client = get_slack_client(configs=configs)

    print("Sending test message ...")
    client.chat_postMessage(
        channel=configs["channel_id"], text=message, unfurl_links=False
    )


# upload_and_then_share_file = client.files_upload_v2(
#     channel="C123456789",
#     title="Test text data",
#     filename="test.txt",
#     content="Hi there! This is a text file!",
#     initial_comment="Here is the file:",
# )


if __name__ == '__main__':
    test_write()