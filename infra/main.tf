# One spot GPU machine for short tests of the PianoFi Live engine. See RUN.md step 9.
#
# It is a test box, not the final service: nothing here touches the PianoFi
# cluster, its load balancer or its data. `terraform destroy` removes all of it.
#
# Applied and destroyed for two test runs, on 2026-11-06 and 2026-11-07.

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
      Purpose   = "gpu-test"
      ManagedBy = "terraform"
    }
  }
}

# Amazon's ECS GPU image: Linux with the NVIDIA driver and Docker already set up.
data "aws_ssm_parameter" "gpu_ami" {
  name = var.ami_ssm_parameter
}

# Not every zone has every GPU type, so ask which zones offer this one.
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

# The account's default network in that zone.
data "aws_subnet" "default" {
  availability_zone = local.zone
  default_for_az    = true
}

# Only SSH, and only from one address. Nothing else can reach the machine.
resource "aws_security_group" "ssh" {
  name_prefix = "pianofi-live-gpu-test-"
  description = "SSH from one address for the PianoFi Live GPU test"
  vpc_id      = data.aws_subnet.default.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  egress {
    description = "Outbound, to download Docker images and ONNX Runtime"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_key_pair" "test" {
  key_name_prefix = "pianofi-live-gpu-test-"
  public_key      = file(pathexpand(var.ssh_public_key_path))
}

resource "aws_instance" "gpu" {
  ami                         = nonsensitive(data.aws_ssm_parameter.gpu_ami.value)
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnet.default.id
  vpc_security_group_ids      = [aws_security_group.ssh.id]
  key_name                    = aws_key_pair.test.key_name
  associate_public_ip_address = true

  # Spot: spare capacity at a lower price. AWS can take it back with 2 minutes'
  # notice, which is fine for a test. "one-time" means it is not restarted.
  instance_market_options {
    market_type = "spot"
    spot_options {
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

  # Safety net for a forgotten machine: it shuts itself down after max_minutes.
  # A one-time spot machine cannot be stopped, so shutting down deletes it (and
  # its disk). AWS refuses instance_initiated_shutdown_behavior on spot, so it
  # is not set here.
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

  # Spot can have no capacity. Give up after this long instead of retrying for ever.
  timeouts {
    create = "5m"
  }

  tags = {
    Name = "pianofi-live-gpu-test"
  }
}
