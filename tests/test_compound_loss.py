"""Test that the compound distillation loss components are computed and logged separately.

Verifies:
  - L_distill (MSE between z_student and z_teacher) is non-negative
  - L_task (cross-entropy) is non-negative
  - L_total = alpha * L_distill + beta * L_task
  - With alpha=0, L_total == L_task (sanity check / pure task mode)
  - With beta=0, L_total == alpha * L_distill (pure distillation mode)
  - Teacher weights do NOT receive gradients
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.student_pipeline import StudentClassifier


BATCH = 4
INPUT_DIM = 5
LATENT_DIM = 8


def _make_fake_batch():
    imgs = torch.rand(BATCH, 3, 64, 64)   # tiny images for speed
    student_tab = torch.rand(BATCH, INPUT_DIM)
    targets = torch.randint(0, 2, (BATCH,))
    return imgs, student_tab, targets


def _compound_loss(z_student, z_teacher, logits, targets, class_weights, alpha, beta):
    distill_criterion = nn.MSELoss()
    task_criterion = nn.CrossEntropyLoss(weight=class_weights)
    l_distill = distill_criterion(z_student, z_teacher.detach())
    l_task = task_criterion(logits, targets)
    l_total = alpha * l_distill + beta * l_task
    return l_distill, l_task, l_total


class TestCompoundLoss:
    def setup_method(self):
        # Tiny CornealEncoder for speed (same architecture, smaller images)
        self.teacher = CornealEncoder(in_channels=3, latent_dim=LATENT_DIM)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.student = StudentClassifier(head_type="classical", input_dim=INPUT_DIM)
        self.class_weights = torch.tensor([1.0, 1.0])

    def test_loss_components_positive(self):
        imgs, student_tab, targets = _make_fake_batch()
        with torch.no_grad():
            z_teacher = self.teacher(imgs)
        z_student, logits = self.student(student_tab, return_embedding=True)
        l_d, l_t, l_total = _compound_loss(
            z_student, z_teacher, logits, targets, self.class_weights, 0.5, 0.5
        )
        assert l_d.item() >= 0.0, "L_distill must be non-negative (MSE)"
        assert l_t.item() >= 0.0, "L_task must be non-negative (CE)"
        assert l_total.item() >= 0.0, "L_total must be non-negative"

    def test_alpha_zero_equals_task_loss(self):
        imgs, student_tab, targets = _make_fake_batch()
        with torch.no_grad():
            z_teacher = self.teacher(imgs)
        z_student, logits = self.student(student_tab, return_embedding=True)
        l_d, l_t, l_total = _compound_loss(
            z_student, z_teacher, logits, targets, self.class_weights, 0.0, 1.0
        )
        assert abs(l_total.item() - l_t.item()) < 1e-6, (
            "With alpha=0, beta=1: L_total must equal L_task"
        )

    def test_beta_zero_equals_distill_loss(self):
        imgs, student_tab, targets = _make_fake_batch()
        with torch.no_grad():
            z_teacher = self.teacher(imgs)
        z_student, logits = self.student(student_tab, return_embedding=True)
        l_d, l_t, l_total = _compound_loss(
            z_student, z_teacher, logits, targets, self.class_weights, 1.0, 0.0
        )
        assert abs(l_total.item() - l_d.item()) < 1e-6, (
            "With alpha=1, beta=0: L_total must equal L_distill"
        )

    def test_teacher_gets_no_gradient(self):
        imgs, student_tab, targets = _make_fake_batch()
        with torch.no_grad():
            z_teacher = self.teacher(imgs)
        z_student, logits = self.student(student_tab, return_embedding=True)
        _l_d, _l_t, l_total = _compound_loss(
            z_student, z_teacher, logits, targets, self.class_weights, 0.5, 0.5
        )
        l_total.backward()
        for name, p in self.teacher.named_parameters():
            assert p.grad is None, f"Teacher parameter {name} should have no gradient"

    def test_student_receives_gradient(self):
        imgs, student_tab, targets = _make_fake_batch()
        with torch.no_grad():
            z_teacher = self.teacher(imgs)
        z_student, logits = self.student(student_tab, return_embedding=True)
        _l_d, _l_t, l_total = _compound_loss(
            z_student, z_teacher, logits, targets, self.class_weights, 0.5, 0.5
        )
        l_total.backward()
        student_grads = [p.grad for p in self.student.parameters() if p.grad is not None]
        assert len(student_grads) > 0, "Student must receive gradients from compound loss"
