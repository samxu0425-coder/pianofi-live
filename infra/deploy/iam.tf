# The front may start the GPU box (the "Start the engine" button on the offline page) and
# nothing else: one role, one policy, scoped to that one instance. Describe has no
# per-instance scoping in IAM, so it is allowed on all; it only reads state.

resource "aws_iam_role" "front" {
  name_prefix = "pianofi-live-front-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "front_start_gpu" {
  name = "start-the-gpu-box"
  role = aws_iam_role.front.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "ec2:StartInstances"
        Resource = aws_instance.gpu.arn
      },
      {
        Effect   = "Allow"
        Action   = "ec2:DescribeInstances"
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_instance_profile" "front" {
  name_prefix = "pianofi-live-front-"
  role        = aws_iam_role.front.name
}
