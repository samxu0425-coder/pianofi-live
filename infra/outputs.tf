output "instance_id" {
  value = aws_instance.gpu.id
}

output "zone" {
  value = aws_instance.gpu.availability_zone
}

output "public_ip" {
  value = aws_instance.gpu.public_ip
}

output "ssh" {
  description = "Log in with this."
  value       = "ssh -i ${trimsuffix(var.ssh_public_key_path, ".pub")} ec2-user@${aws_instance.gpu.public_ip}"
}
