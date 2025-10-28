# photon-mosaic-roicat
This is an enxtension for [photon-mosaic](https://github.com/neuroinformatics-unit/photon-mosaic), which also adds support for [ROICaT](https://github.com/RichieHakim/ROICaT). This repository is heavily adapted from prior work by [Athina Apostolelli **@AthinaApostolelli**](https://www.sainsburywellcome.org/web/people/athina-apostolelli).

## Getting started
```sh
conda activate photon-mosaic-dev # We will use the same env as photon-mosaic

# ROICaT
pip install roicat[all]
pip install git+https://github.com/RichieHakim/roiextractors
pip uninstall torch # because roicat installation will install non-CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# notification
pip install git+https://github.com/neuroinformatics-unit/swc-slack # it works well with any Slack

# Install other dependencies
pip install submitit fpdf2
pip install -e.
```

## Daily usage

When running only photon-mosaic
```sh
cd slurm_jobs
sbatch run_pm.sh
```

When running photon-mosaic + ROICaT
```sh
cd slurm_jobs
sbatch run_pm_roicat.sh
```

## Use Slack to notify your and your colleages

1. First make your own slack app from [here](https://api.slack.com/apps). You should also take a look at [this page](https://api.slack.com/apis/connections/socket) and [some of official examples](https://github.com/slack-samples/bolt-python-starter-template/tree/main) to understand how Slack bots work.

```json
{
  "_metadata": {
      "major_version": 1
  },
  "display_information": {
      "name": "2p-313"
  },
  "features": {
      "app_home": {
          "home_tab_enabled": true,
          "messages_tab_enabled": true,
          "messages_tab_read_only_enabled": true
      },
      "bot_user": {
          "display_name": "2p-313",
          "always_online": true
      }
  },
  "oauth_config": {
      "scopes": {
          "bot": [
              "channels:history",
              "chat:write",
              "files:write",
              "channels:join",
              "groups:write",
              "im:write",
              "mpim:write"
          ]
      }
  },
  "settings": {
      "event_subscriptions": {
          "bot_events": [
              "message.channels"
          ]
      },
      "interactivity": {
          "is_enabled": false
      },
      "org_deploy_enabled": true,
      "socket_mode_enabled": true,
      "token_rotation_enabled": false
  }
}
```

2. Create a Slack channel #sumiya-2p and copy its ID. Also add your App **2p-313** (Here is a link to [the app](https://api.slack.com/apps/A09LT1WU9SP). Contact Sumiya if you need access). 

3. You can run `python notification/slack_bot.py` to send a test message.