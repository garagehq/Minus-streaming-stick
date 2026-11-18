#!/bin/sh
# Turn off the ROCK 5B+'s own status LEDs (minus-board-leds.service, at boot).
#
# Only LEDs owned by the kernel's gpio-leds driver are touched: on the
# ROCK 5B+ that is user-led2, the blinking status/activity LED (heartbeat
# trigger by default). The power LED is wired to the power rail and cannot
# be switched off in software. The red mute LED on header pin 40 is a plain
# GPIO line driven by Minus (src/mute_led.py), not a gpio-leds LED, so it is
# never affected.
for led in /sys/class/leds/*; do
    case "$(readlink -f "$led")" in
        */gpio-leds/*)
            echo none > "$led/trigger" 2>/dev/null
            echo 0 > "$led/brightness" 2>/dev/null
            echo "board LED off: $(basename "$led")"
            ;;
    esac
done
