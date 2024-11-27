import cv2 as cv
from PIL import Image, ImageDraw, ImageFont

# ChArUco board variables
CHARUCOBOARD_ROWCOUNT = 8
CHARUCOBOARD_COLCOUNT = 11
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
squareLength = 0.02
markerLength = 0.015

# Create the ChArUco board
CHARUCO_BOARD = cv.aruco.CharucoBoard(
    (CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
    squareLength=squareLength,
    markerLength=markerLength,
    dictionary=ARUCO_DICT,
)

# A4 size dimensions at 300 DPI
a4_width, a4_height = (3508, 2480)
margin = 100  # Margin from each side in pixels

# Create an image to draw the ChArUco board
board_image = CHARUCO_BOARD.generateImage(
    (a4_width - margin, int((a4_height * 0.95) - margin)), marginSize=margin
)

# Convert to RGB mode expected by PIL
board_image_rgb = cv.cvtColor(board_image, cv.COLOR_GRAY2RGB)
# Convert OpenCV image to PIL format
pil_image = Image.fromarray(board_image_rgb)

# Define font size and font (PIL's default font is used if a TTF file is not available)
font_size = 60
try:
    # Attempt to load a common TTF font (ensure the path is correct for your environment)
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    # Use default font if the specified TTF font is not available
    font = ImageFont.load_default()

# Define text to be added as a single line
text = (
    f"ROW: {CHARUCOBOARD_ROWCOUNT} | "
    f"COL: {CHARUCOBOARD_COLCOUNT} | "
    f"ARUCO_DICT: DICT_4X4_250 | "
    f"square: {squareLength*1000} mm | "
    f"marker: {markerLength*1000} mm"
)

# Calculate text position
text_x = margin
text_y = int(a4_height * 0.9) + margin

# Create a new A4-sized image
final_image = Image.new("RGB", (a4_width, a4_height), "white")
final_image.paste(pil_image, (margin, margin))  # Paste the board leaving a top margin

# Update drawing context to the new image
draw_final = ImageDraw.Draw(final_image)

# Draw the single line of text on the new image
draw_final.text((text_x, text_y), text, font=font, fill="black")

# Save as PDF
final_image.save("charuco_board.pdf", "PDF", resolution=300.0)
