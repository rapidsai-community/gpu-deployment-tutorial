# Scipy 2026 GPU Deployment tutorial

This tutorial will discuss how to get your own GPUs on the cloud in more general terms. In order to dig into some of the
things we will be learning, we will launch a VM through the [NVIDIA Brev](https://brev.nvidia.com) portal.

## Getting set up with Brev

### In person

- Follow the invite link provided by the tutorial teachers to get invited to
  the tutorial `gpu-deployment-scipy26` organization

### On demand

- Sign in to or register an account at [NVIDIA Brev](https://brev.nvidia.com)
- Ensure you have some credits or an active account

#### Launching a Brev VM

- Under "GPUs" select "Create Environment"
- Choose a GPU type that costs <$1/hour (e.g an L4)
- Choose GCP or AWS as the provider
  - They have the most availability
- Give your VM a name
- Press Deploy
- Wait for the VM to be "Running" and the software environment to finish "Building"

#### Connecting to your VM

Once your VM is deployed, follow the Brev access instructions provided for your instance. The connection instructions will vary depending on your operating system. For example, on macOS you would:

- Install the `brev` CLI
  - `brew install brevdev/homebrew-brev/brev`
- Login to your account
  - `brev login`
- Make sure the right organization is set.
- Connect via SSH
  - `brev ls` to list your VMs
  - `brev shell <your vm name>` to connect via SSH

```[!NOTE]
Protip: You can also run `brev refresh` which adds all of your VMs to your `~/.ssh/config` which allows you to run `ssh <your vm name>` directly. This is especially helpful if you want to use `scp` or `rsync` to copy data and files between your local machine and your VM
```

For Linux and Windows instructions check the [brev-cli install documentation](https://docs.nvidia.com/brev/latest/cli/getting-started#installation)
