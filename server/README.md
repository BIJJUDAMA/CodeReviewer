---
title: CoderReviewer
sdk: docker
emoji: 🛡️
colorFrom: blue
colorTo: gray
pinned: false
---

# OpenEnv Code Review Server

This is the environment server for the OpenEnv Code Review competition. It provides a standardized interface for AI agents to perform code review tasks.

## Features
- Deterministic evaluation using AST and test-case execution.
- Tiered reward shaping for improved learning signal.
- Support for security auditing and performance refactoring tasks.

## Interface
- **Reset**: POST /reset
- **Step**: POST /step
- **Docs**: /docs
