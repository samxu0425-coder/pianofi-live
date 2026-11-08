# One spot GPU machine that serves PianoFi Live to a couple of users. RUN.md step 11.
#
# Two machines: the GPU box here (spot, persistent, stops itself after max_minutes) and
# the always-on front in front.tf, which holds the fixed address, HTTPS and the password,
# and shows an "engine offline" page while the GPU box is down. `terraform destroy`
# removes both. Separate state from the test box in ../, so the two never touch each other.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project   = "pianofi-live"
      Purpose   = "service"
      ManagedBy = "terraform"
    }
  }
}

data "aws_ssm_parameter" "gpu_ami" {
  name = var.ami_ssm_parameter
}

data "aws_ec2_instance_type_offerings" "gpu" {
  location_type = "availability-zone"
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

locals {
  zone = var.availability_zone != "" ? var.availability_zone : sort(data.aws_ec2_instance_type_offerings.gpu.locations)[0]
}

data "aws_subnet" "default" {
  availability_zone = local.zone
  default_for_az    = true
}

resource "aws_security_group" "web" {
  name_prefix = "pianofi-live-service-"
  # The description must stay as first created: changing it replaces the group.
  description = "PianoFi Live: HTTPS for users, SSH from one address"
  vpc_id      = data.aws_subnet.default.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }
  ingress {
    description     = "page and events, from the front machine only"
    from_port       = 8000
    to_port         = 8001
    protocol        = "tcp"
    security_groups = [aws_security_group.front.id]
  }
  egress {
    description = "Outbound, for Docker images"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_key_pair" "service" {
  key_name_prefix = "pianofi-live-service-"
  public_key      = file(pathexpand(var.ssh_public_key_path))
}

# A fixed address, so the hostname (and its certificate) survive a replacement machine.
resource "aws_eip" "service" {
  domain = "vpc"
}

resource "aws_instance" "gpu" {
  ami                    = nonsensitive(data.aws_ssm_parameter.gpu_ami.value)
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnet.default.id
  vpc_security_group_ids = [aws_security_group.web.id]
  key_name               = aws_key_pair.service.key_name

  # Persistent spot: when AWS takes the capacity back the machine is stopped, not
  # deleted, and AWS starts it again when capacity returns. The disk, the address and
  # the containers (restart policy) survive, so the site comes back on its own. The
  # first machine, a one-time request in us-east-1a, was reclaimed 8 minutes in.
  instance_market_options {
    market_type = "spot"
    spot_options {
      spot_instance_type             = "persistent"
      instance_interruption_behavior = "stop"
    }
  }

  # Safety net: the machine stops itself after max_minutes (a stopped machine costs
  # only its disk, about $0.20 a day). Runs on the first boot only.
  user_data = <<-EOT
    #!/bin/bash
    shutdown -h +${var.max_minutes}
  EOT

  root_block_device {
    volume_size           = var.disk_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  metadata_options {
    http_tokens = "required"
  }

  tags = {
    Name = "pianofi-live-service"
  }
}

# The address belongs to the front machine (front.tf); the GPU box sits behind it.
resource "aws_eip_association" "service" {
  instance_id   = aws_instance.front.id
  allocation_id = aws_eip.service.id
}
