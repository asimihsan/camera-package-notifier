#!/usr/bin/env node
import * as cdk from '@aws-cdk/core';
import { TrainingStack } from '../lib/training-stack';

const environment = {
    account: '519160639284',
    region: 'us-west-2',
}

const app = new cdk.App();
new TrainingStack(app, 'camera-package-notifier-TrainingStack', { env: environment });
