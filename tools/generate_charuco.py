import cv2 as cv
from PIL import Image, ImageDraw, ImageFont


def generate_charuco_board(
    CHARUCOBOARD_ROWCOUNT=15,
    CHARUCOBOARD_COLCOUNT=24,
    dictionary_id=cv.aruco.DICT_5X5_1000,
    DPI=300,
    paper_size="A3",
):
    # Create a mapping from dictionary IDs to their string names
    aruco_dict_list = [attr for attr in dir(cv.aruco) if attr.startswith("DICT_")]
    dict_name_to_value = {name: getattr(cv.aruco, name) for name in aruco_dict_list}
    dict_value_to_name = {value: name for name, value in dict_name_to_value.items()}

    # Get the specified ArUco dictionary
    ARUCO_DICT = cv.aruco.getPredefinedDictionary(dictionary_id)

    # ChArUco board square and marker sizes in meters
    squareLength = 0.022  # Default square length in meters
    markerLength = 0.018  # Default marker length in meters

    # Printing parameters
    pixels_per_meter = (
        DPI / 0.0254
    )  # Number of pixels in one meter at the specified DPI

    # Compute square and marker sizes in pixels
    squareLength_px = squareLength * pixels_per_meter
    markerLength_px = markerLength * pixels_per_meter

    # Compute board size in pixels
    board_width_px = CHARUCOBOARD_COLCOUNT * squareLength_px
    board_height_px = CHARUCOBOARD_ROWCOUNT * squareLength_px

    # Define page sizes at the specified DPI (landscape)
    if paper_size == "A4":
        page_width_mm, page_height_mm = (297, 210)
    elif paper_size == "A3":
        page_width_mm, page_height_mm = (420, 297)
    elif paper_size == "A2":
        page_width_mm, page_height_mm = (594, 420)
    else:
        raise ValueError("Unsupported paper size. Please use 'A4', 'A3', or 'A2'.")

    # Convert page size from mm to pixels
    page_width_px = int(page_width_mm * DPI / 25.4)
    page_height_px = int(page_height_mm * DPI / 25.4)

    # Margins in mm and pixels
    margin_mm = 25  # Adjust margins as needed
    margin_px = int(margin_mm * DPI / 25.4)

    # Calculate text area height in pixels
    font_size = 60  # Define font size in pixels
    text_area_height_px = font_size + margin_px

    # Check if the board fits on the page with margins
    max_board_width_px = page_width_px - 2 * margin_px
    max_board_height_px = page_height_px - 2 * margin_px - text_area_height_px

    # Adjust square length if necessary
    if board_width_px > max_board_width_px or board_height_px > max_board_height_px:
        scale_factor = min(
            max_board_width_px / board_width_px, max_board_height_px / board_height_px
        )
        squareLength_px *= scale_factor
        markerLength_px *= scale_factor
        board_width_px = int(board_width_px * scale_factor)
        board_height_px = int(board_height_px * scale_factor)
        squareLength = squareLength_px / pixels_per_meter
        markerLength = markerLength_px / pixels_per_meter

    # Create the ChArUco board with adjusted sizes
    CHARUCO_BOARD = cv.aruco.CharucoBoard(
        (CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
        squareLength=squareLength,
        markerLength=markerLength,
        dictionary=ARUCO_DICT,
    )

    # Generate the board image
    board_image = CHARUCO_BOARD.generateImage(
        (int(board_width_px), int(board_height_px)), marginSize=0
    )

    # Convert to PIL image
    board_image_rgb = cv.cvtColor(board_image, cv.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(board_image_rgb)

    # Define font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Get the dictionary name
    aruco_dict_name = dict_value_to_name.get(dictionary_id, "Unknown Dictionary")

    # Define text to be added
    text = (
        "Charuco Target | "
        f"ROW: {CHARUCOBOARD_ROWCOUNT} | "
        f"COL: {CHARUCOBOARD_COLCOUNT} | "
        f"ARUCO_DICT: {aruco_dict_name} | "
        f"square: {squareLength*1000:.2f} mm | "
        f"marker: {markerLength*1000:.2f} mm | "
        f"DPI: {DPI}"
    )

    # Create the final image with page size
    final_image = Image.new("RGB", (page_width_px, page_height_px), "white")

    # Paste the board image onto the final image with margins
    final_image.paste(pil_image, (margin_px, margin_px))

    # Draw the text on the final image
    draw_final = ImageDraw.Draw(final_image)
    # Get the bounding box of the text
    bbox = draw_final.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    # Calculate text position
    text_x = margin_px
    text_y = page_height_px - margin_px - text_height
    # Draw the text
    draw_final.text((text_x, text_y), text, font=font, fill="black")

    # Save as PDF
    filename = f"charuco_board_{paper_size}.pdf"
    final_image.save(filename, "PDF", resolution=DPI)
    print(f"ChArUco board saved to {filename}")


generate_charuco_board(
    CHARUCOBOARD_ROWCOUNT=15,
    CHARUCOBOARD_COLCOUNT=24,
    dictionary_id=cv.aruco.DICT_5X5_1000,
    DPI=300,
    paper_size="A2",
)
