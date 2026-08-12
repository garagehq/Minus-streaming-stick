# MARISOL - Minus AI Research and Integration Support Operator Liaison

## Overview

MARISOL is an AI assistant persona that helps maintain and develop the Minus project. This document provides context for AI agents working on this codebase.

## Project Summary

**Minus** is an HDMI passthrough device that blocks ads in real-time using dual NPU machine learning:
- **Primary NPU**: RK3588 running PaddleOCR for text detection (~400ms)
- **Secondary NPU**: Axera LLM 8850 running minus-v0.1 (fine-tuned LFM2.5-VL-450M, fused-prefill, no decode) for visual analysis (~0.37s)

The device sits between a Fire TV and a display, detecting ads and replacing them with a Spanish vocabulary practice overlay.

## Key Architecture Decisions

### Why Dual NPUs?
- OCR is fast but only catches text-based ad indicators
- VLM provides semantic understanding but is slower
- Running both in parallel covers more ad types
- Weighted voting prevents false positives

### Why ustreamer Blocking Mode?
- GStreamer overlays caused 12-second periodic stalls
- ustreamer's MPP encoder composites at 60fps natively
- All blocking done in hardware encoder, not Python
- HTTP API is simple and reliable

### Why Spanish Vocabulary?
- Productive use of blocked time
- Owner is learning Spanish
- 500+ intermediate vocabulary words
- Rotates every 11-15 seconds

## Code Conventions

### File Organization
- `src/` - All Python modules
- `docs/` - Reference documentation
- `tests/` - Test suite
- Root - Entry point, config, install scripts

### Naming Conventions
- Classes: PascalCase (e.g., `DRMAdBlocker`, `FireTVController`)
- Functions/methods: snake_case (e.g., `check_skip_opportunity`)
- Private methods: prefix with `_` (e.g., `_restart_pipeline`)
- Constants: UPPER_SNAKE_CASE (e.g., `KEY_CODES`, `AD_KEYWORDS`)

### Threading
- All shared state protected by `threading.Lock()`
- Use `threading.Event()` for signaling stop conditions
- Watchdogs use atomic flags
- GStreamer callbacks run in their own threads

### Error Handling
- Log errors with context
- Use exponential backoff for retries
- Graceful degradation (VLM failures → OCR-only mode)
- Never crash the main loop

## Common Tasks

### Adding a New API Endpoint

1. Add route to `src/webui.py`
2. Add test to `tests/test_modules.py`
3. Update documentation

```python
@app.route('/api/new-endpoint', methods=['POST'])
def api_new_endpoint():
    # Implementation
    return jsonify({'success': True})
```

### Adding a New Test

1. Find or create appropriate test class in `tests/test_modules.py`
2. Add test method with `test_` prefix
3. Use mocks for external dependencies
4. Run `python3 tests/test_modules.py` to verify

### Modifying Blocking Overlay

1. Text/colors: Modify `_get_blocking_text()` in `src/ad_blocker.py`
2. Layout: Adjust ustreamer API call parameters
3. Preview: Modify `_preview_enabled` and related settings
4. Test with `/api/test/trigger-block` endpoint

### Adding Vocabulary

1. Edit `src/vocabulary.py`
2. Add tuple: `("spanish", "pronunciation", "english", "example sentence")`
3. Run vocabulary tests to verify format

## Important Files

### Must Read Before Changes

| File | Contains |
|------|----------|
| `CLAUDE.md` | Complete development notes |
| `docs/ARCHITECTURE.md` | System architecture |
| `docs/AESTHETICS.md` | Visual design guide |
| `tests/test_modules.py` | Test suite |

### Critical Code Paths

| Path | Impact |
|------|--------|
| `minus.py:_detection_loop()` | Main ad detection logic |
| `src/ad_blocker.py:show()/hide()` | Blocking state changes |
| `src/audio.py:mute()/unmute()` | Audio control |
| `src/health.py:_check_loop()` | System health monitoring |

## Testing

### Running Tests

```bash
python3 tests/test_modules.py  # 242+ tests
```

### Test Categories

- **Unit tests**: Individual function/method tests
- **Integration tests**: Cross-module interaction
- **Mock tests**: Tests with mocked hardware dependencies

### Writing Good Tests

1. Mock external dependencies (NPU, hardware, network)
2. Test error conditions, not just happy path
3. Use descriptive names (`test_fire_tv_reconnect_on_disconnect`)
4. Check return values AND side effects

## Debugging

### Common Issues

**Video pipeline stalls:**
- Check ustreamer health: `curl http://localhost:9090/state`
- Restart video: `curl -X POST http://localhost/api/video/restart`
- Check FPS in logs

**VLM not responding:**
- Check Axera card: `axcl_smi`
- VLM may degrade to OCR-only after 5 failures
- Restart service to reload VLM

**Fire TV disconnected:**
- Check status: `curl http://localhost/api/firetv/status`
- Auto-reconnect should recover within 30s
- May need to re-authorize ADB on TV

### Log Locations

- Service logs: `journalctl -u minus -f`
- Application log: `/tmp/minus.log`
- Web UI: View in Settings tab

## Git Workflow

### Commit Messages

- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Keep messages concise and descriptive
- NO AI attribution lines (no Co-Authored-By, no Generated with Claude)

### Branches

- `main` - Production code
- `feature/*` - New features
- `fix/*` - Bug fixes

### Pull Requests

- Create from feature branches
- Include test changes
- Update documentation if needed
- Use `gh pr create` for convenience

## Performance Considerations

### Hot Paths

- OCR runs every ~500ms
- VLM runs every ~1s
- Blocking API calls during state changes
- FPS probe callback on every frame

### Memory Management

- Explicit cleanup of RKNN outputs
- Periodic `gc.collect()` in workers
- Health monitor triggers cleanup at 90% memory
- Frame buffers cleared during critical events

### Threading Concerns

- Lock contention in status methods
- HTTP timeouts affect detection loop
- Watchdogs must not block main thread
- GStreamer callbacks run async

## Contact

For questions about the codebase, consult:
1. `CLAUDE.md` - Comprehensive development notes
2. `docs/` - Reference documentation
3. Code comments - Implementation details
4. Test suite - Expected behavior examples
