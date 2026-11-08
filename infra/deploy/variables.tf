variable "region" {
  description = "AWS region. The original PianoFi runs in us-east-1."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU machine type. g4dn.xlarge = 1 NVIDIA T4, enough for 2 engines at once."
  type        = string
  default     = "g4dn.xlarge"
}

variable "availability_zone" {
  description = "Zone to use. Empty = the first zone that offers instance_type. On 2026-11-07 us-east-1b had no spot T4s, 1a reclaimed one after 8 minutes; 1d's default subnet is private (NAT route); 1c and 1f scored 9/10 (aws ec2 get-spot-placement-scores)."
  type        = string
  default     = "us-east-1f"
}

variable "ami_ssm_parameter" {
  description = "Public SSM parameter that holds the id of Amazon's current ECS GPU image (NVIDIA driver and Docker included)."
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
  description = "Disk size. The CUDA image, the build and uploaded songs need room."
  type        = number
  default     = 60
}

variable "max_minutes" {
  description = "The machine stops itself this many minutes after its first boot. 480 = 8 hours, about $2 of T4 time. Start it again from the console or with `aws ec2 start-instances`."
  type        = number
  default     = 480
}

variable "auth_user" {
  description = "The admin's user name at the site's password prompt. The admin can add songs and start the GPU box as often as needed."
  type        = string
  default     = "sam"
}

variable "auth_hash" {
  description = "bcrypt hash of the admin's password, from `docker run --rm caddy:2 caddy hash-password --plaintext '...'`. Kept in terraform.tfvars, which git ignores."
  type        = string
  sensitive   = true
}

variable "guest_user" {
  description = "The guest's user name. A guest can play and transcribe, but not add songs, and can start the GPU box only guest_starts_per_day times a day."
  type        = string
  default     = "guest"
}

variable "guest_hash" {
  description = "bcrypt hash of the guest's password, made like auth_hash."
  type        = string
  sensitive   = true
}

variable "guest_starts_per_day" {
  description = "How many times a day a guest may start the GPU box from the offline page. Each start costs at most idle_minutes of T4 time if nobody plays."
  type        = number
  default     = 3
}

variable "idle_minutes" {
  description = "The GPU box stops itself after this many minutes with no engine running (installed by deploy/run.sh). A stopped box costs only its disk."
  type        = number
  default     = 30
}
