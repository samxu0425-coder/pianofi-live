# The always-on front door: a t4g.nano (about $3 a month) that holds the fixed address and
# runs Caddy (HTTPS, the password). It proxies to the GPU box over the VPC and serves
# deploy/front/offline.html when the box is stopped or reclaimed, instead of going dark.

data "aws_ssm_parameter" "front_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

resource "aws_security_group" "front" {
  name_prefix = "pianofi-live-front-"
  description = "PianoFi Live front: HTTPS for users, SSH from one address"
  vpc_id      = data.aws_subnet.default.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }
  ingress {
    description = "HTTP, redirected to HTTPS and used for the certificate check"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "front" {
  ami                    = nonsensitive(data.aws_ssm_parameter.front_ami.value)
  instance_type          = "t4g.nano"
  subnet_id              = data.aws_subnet.default.id
  vpc_security_group_ids = [aws_security_group.front.id]
  key_name               = aws_key_pair.service.key_name
  iam_instance_profile   = aws_iam_instance_profile.front.name # may start the GPU box, nothing else (iam.tf)

  # Everything the front needs is set up on first boot: the Caddy binary (no Docker: the
  # nano's 512 MB can't install it), the files from deploy/front/ (Caddyfile, offline page,
  # the start handler) and an environment file with the logins and the GPU box's private
  # address; deploy/front/install.sh turns them into two systemd services.
  user_data = templatefile("${path.module}/front_user_data.sh", {
    site          = "${replace(aws_eip.service.public_ip, ".", "-")}.sslip.io"
    engine_ip     = aws_instance.gpu.private_ip
    instance_id   = aws_instance.gpu.id
    region        = var.region
    auth_user     = var.auth_user
    auth_hash     = var.auth_hash
    guest_user    = var.guest_user
    guest_hash    = var.guest_hash
    guest_starts  = var.guest_starts_per_day
    caddy_version = "2.9.1"
    caddyfile     = file("${path.module}/../../deploy/front/Caddyfile")
    offline       = file("${path.module}/../../deploy/front/offline.html")
    start_py      = file("${path.module}/../../deploy/front/start.py")
    install_sh    = file("${path.module}/../../deploy/front/install.sh")
  })
  # Changes to the first-boot script are not applied by replacing the machine (that would
  # cost a new certificate; Let's Encrypt allows 5 a week per name). A changed file in
  # deploy/front/ is shipped by hand: scp the folder and run `sudo bash front/install.sh`
  # (RUN.md step 11). The provider stops and starts the machine for the new user_data;
  # the address and the certificate survive that.
  user_data_replace_on_change = false

  root_block_device {
    volume_size = 8
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required"
  }

  tags = {
    Name = "pianofi-live-front"
  }
}
