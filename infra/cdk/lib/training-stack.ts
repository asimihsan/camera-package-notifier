import * as autoscaling from '@aws-cdk/aws-autoscaling';
import * as cdk from '@aws-cdk/core';
import * as ec2 from '@aws-cdk/aws-ec2';
import * as iam from '@aws-cdk/aws-iam';

export class TrainingStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // NVIDIA Deep Learning AMI v20.11.0-46a68101-e56b-41cd-8e32-631ac6e5d02b
    // See: https://aws.amazon.com/marketplace/pp/NVIDIA-NVIDIA-Deep-Learning-AMI/B076K31M1S
    // const nvidiaAmiMap = {
    //   "us-west-2": "ami-08ec8407396bfed33",
    // };

    const deepLearningAmiAmazonLinux2 = {
      "us-west-2": "ami-01a495658aa5f7930",
    }

    const vpc = new ec2.Vpc(this, 'VPC', {
      maxAzs: 3,
      subnetConfiguration: [
        {
          subnetType: ec2.SubnetType.PUBLIC,
          name: 'DefaultSubnet'
        }
      ]
    });

    const role = new iam.Role(this, 'AmiPreparerRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com')
    });
    role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'));

    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      'sudo yum -y --exclude=kernel\* update',
      'sudo yum install -y zstd zsh curl util-linux-user',
      'sudo yum groupinstall -y "Development Tools"',

      'ZSH=/home/ssm-user sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended',
      'sudo chsh -s $(which zsh) ssm-user',

      'ZSH=/home/ec2-user sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended',
      'sudo chsh -s $(which zsh) ec2-user',
      'sudo mkdir -p /home/ec2-user/.ssh',
      'sudo chmod 700 /home/ec2-user/.ssh',
      'sudo touch /home/ec2-user/.ssh/authorized_keys',
      'sudo chmod 600 /home/ec2-user/.ssh/authorized_keys',
      'echo ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCxrtz0y51DemTwwaJhxZKjT1maXl1k3FQ9TY8RpKDf0FkT0cmerFrtpwgNbje9yR9JETS+cW3xFKM/sFROKyf1JRXRhwz/yeKUxISjq2ozcW9Q+OFqtG626d99+Q8xQ7qmz3IvwSmpCG+KVDiiZKlE7odVygFUx2MVOr6G73jAjEfluVDIAZe5fGSJ4Nr38xZ7jJG72ht99iJw/11qcZo31fWHZ/ukoloEigRTrPtRepAcSGn3LJJVyYmHfjonimjqUHCBBkwsiAHa5MMLVhQK6qDMu2tfIyZNAzJQmwv/Y6cO40WhtUvSv/+AJ0kTjqLxMhRdmpbEscWHbjahHl8++AJ9yR/b+Wro0s9EWNrhsBN+3iqZp7XofD83c0YwbbAbbtO6HEq4v/JKCEq+0I7aVtwp+BfWOcjaA/yEUo3cUr0SSOdigeLdUN7jfLjppBvFgm71FP81LGpOTk+pieQmK9qkcMF1IgemzCBrBU8RlKSHgXDsPQFCEhA0B2m8iU6bfNEHcBqRnWzw3+G+9d4p7AnEq+POc3ER9PYGWypfhG2gtaX/ZGdtJrIDFCoA0rK7HBT/KO98Ec/y27xDn8P62Q8OPk/0F//4/wa+ilgMoJkfEIjzQDeUBkc3d1R2mvlSFnSMz7n/+8vdjhzurfW41QnjmLRup4utqRsUB8MCGQ== | sudo tee -a /home/ec2-user/.ssh/authorized_keys',
      'echo source activate tensorflow2_latest_p37 | sudo tee -a /home/ec2-user/.zshrc',
      '/home/ec2-user/anaconda3/bin/conda config --set auto_activate_base true',
      '/home/ec2-user/anaconda3/envs/tensorflow2_latest_p37/bin/pip install black ipdb mypy numpy oauthlib opencv-contrib-python-headless pillow pylint requests retry ring_doorbell scikit-image scikit-learn scipy twilio zstandard',
      '/home/ec2-user/anaconda3/envs/tensorflow2_latest_p37/bin/python -c "import tensorflow as tf; print(tf.config.experimental.list_physical_devices(\"GPU\"))"',
    );

    const trainingAsg = new autoscaling.AutoScalingGroup(this, 'AmiPreparerAsg', {
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.P2, ec2.InstanceSize.XLARGE),
      machineImage: ec2.MachineImage.genericLinux(deepLearningAmiAmazonLinux2),
      minCapacity: 0,
      desiredCapacity: 1,
      maxCapacity: 2,
      spotPrice: "0.5",
      role: role,
      userData: userData,
      updatePolicy: autoscaling.UpdatePolicy.replacingUpdate(),
    });
    trainingAsg.connections.allowFromAnyIpv4(ec2.Port.tcp(22));

  }
}
