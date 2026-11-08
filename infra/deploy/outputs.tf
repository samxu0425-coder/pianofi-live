output "instance_id" {
  value = aws_instance.gpu.id
}

output "public_ip" {
  description = "The fixed address, held by the front machine."
  value       = aws_eip.service.public_ip
}

output "site" {
  description = "Hostname for the page: sslip.io resolves it to the address, so no domain is needed."
  value       = "${replace(aws_eip.service.public_ip, ".", "-")}.sslip.io"
}

output "ssh_front" {
  description = "Log in to the front machine with this."
  value       = "ssh -i ${trimsuffix(var.ssh_public_key_path, ".pub")} ec2-user@${aws_eip.service.public_ip}"
}

output "gpu_public_ip" {
  description = "The GPU box's own (changing) public address, for SSH and uploads."
  value       = aws_instance.gpu.public_ip
}

output "gpu_private_ip" {
  value = aws_instance.gpu.private_ip
}
