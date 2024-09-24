import win32api
import win32con
import time

# Get the screen width
screen_width = 2560  # Width

# Function to move the mouse right by half the screen width
def move_mouse_half_width():
    # Get current mouse position
    current_pos = win32api.GetCursorPos()

    # Calculate the movement (half the screen width)
    dx = int(screen_width / 2)

    # Move the mouse relative to its current position
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, 0, 0, 0)

# Example usage
time.sleep(2)  # Wait 2 seconds to switch to the game
move_mouse_half_width()