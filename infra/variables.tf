variable "region" {
  description = "AWS region. The original PianoFi runs in us-east-1."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU machine type. g4dn.xlarge = 1 NVIDIA T4 (16 GB), 4 vCPUs, 16 GB RAM."
  type        = string
  default     = "g4dn.xlarge"
}

variable "availability_zone" {
  description = "Zone to use, e.g. us-east-1b. Empty = the first zone that offers instance_type. Set it if spot has no capacity in that zone."
  type        = string
  default     = ""
}

variable "ami_ssm_parameter" {
  description = "Public SSM parameter that holds the id of Amazon's current ECS GPU image."
  type        = string
  default     = "/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id"
}

variable "my_ip_cidr" {
  description = "The one address allowed to SSH in, as x.x.x.x/32. No default on purpose."
  type        = string

  validation {
    condition     = can(cidrhost(var.my_ip_cidr, 0)) && var.my_ip_cidr != "0.0.0.0/0"
    error_message = "Use your own public address followed by /32, not 0.0.0.0/0."
  }
}

variable "ssh_public_key_path" {
  description = "Public half of the SSH key used to log in."
  type        = string
  default     = "~/.ssh/pianofi_live.pub"
}

variable "disk_gb" {
  description = "Disk size. The CUDA Docker image and the build need room."
  type        = number
  default     = 60
}

variable "max_minutes" {
  description = "The machine shuts itself down and is deleted this many minutes after it starts."
  type        = number
  default     = 120
}
