from PIL import Image

def compose(bg_img, cloth_img):
    bg = bg_img.copy().convert("RGBA")
    fg = cloth_img.convert("RGBA")

    x = (bg.width - fg.width) // 2
    y = (bg.height - fg.height) // 2
    bg.paste(fg, (x, y), fg)
    return bg
