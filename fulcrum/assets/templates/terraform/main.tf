terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "GCP Region"
}

variable "zone" {
  type        = string
  default     = "us-central1-a"
  description = "GCP Zone"
}

variable "model_id" {
  type        = string
  default     = "google/gemma-3-12b-it"
  description = "Hugging Face model ID"
}

variable "machine_type" {
  type        = string
  default     = "g2-standard-8"
  description = "GCE Machine type"
}

variable "gpu_type" {
  type        = string
  default     = "nvidia-l4"
  description = "Accelerator type"
}

variable "gpu_count" {
  type        = number
  default     = 1
  description = "Number of GPUs"
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_firewall" "allow_fulcrum_vllm" {
  name    = "fulcrum-allow-vllm-8000"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["fulcrum-vllm-node"]
}

resource "google_compute_instance" "fulcrum_instance" {
  name         = "fulcrum-inference-node"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["fulcrum-vllm-node"]

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/common-cu124-debian-11"
      size  = 200
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = file("${path.module}/../startup-script.sh")
}

output "instance_ip" {
  value       = google_compute_instance.fulcrum_instance.network_interface[0].access_config[0].nat_ip
  description = "Public IP of the deployed inference node"
}

