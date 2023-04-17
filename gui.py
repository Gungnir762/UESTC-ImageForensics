import PySimpleGUI as sg
import io
from PIL import Image
from forensics import get_img_bin_io


def load_image(img_path):
    with open(img_path, 'rb') as f:
        img_data = f.read()
    return Image.open(io.BytesIO(img_data))


def save_image_to_bytesio(image, img_format='PNG'):
    bio = io.BytesIO()
    image.save(bio, format=img_format)
    return bio.getvalue()


def show_img(image):
    # 计算图片缩放比例
    img_width, img_height = image.size
    window_width, window_height = window.Size
    scale = min(window_width / img_width, window_height / img_height)

    # 调整图片大小并更新到窗口中
    new_img_height = int(img_height * scale) - 80
    new_img_width = int(new_img_height / img_height * img_width)

    image = image.resize((new_img_width, new_img_height))

    window['-IMAGE1-'].update(data=save_image_to_bytesio(image))

    default_img_path = './default.jpg'
    default_image = load_image(default_img_path)
    img_data = save_image_to_bytesio(default_image)
    window['-IMAGE2-'].update(data=img_data)

    window.size = (new_img_width * 2 + 80, new_img_height + 80)
    window.refresh()  # 强制刷新整个窗口


if __name__ == '__main__':
    # 定义GUI布局
    layout = [
        [sg.Text('选择图片：', font=('Arial', 16)), sg.Input(key='-FILE-', size=(20, 1)),
         sg.FileBrowse(font=('Arial', 12))],
        [sg.Image(key='-IMAGE1-'), sg.VerticalSeparator(), sg.Image(key='-IMAGE2-')],
        [sg.Button('上传', size=(10, 2), button_color=('white', '#4CAF50'), font=('Arial', 14)),
         sg.Button('退出', size=(10, 2), button_color=('white', '#F44336'), font=('Arial', 14))],
    ]

    # 创建GUI窗口
    window = sg.Window('图片取证器', layout, size=(800, 600), resizable=True, font=('Arial', 14),
                       element_justification='center', finalize=True)

    while True:
        event, values = window.read()

        # 处理事件
        if event == '上传':
            img_path = values['-FILE-']
            if img_path:
                image = load_image(img_path)
                show_img(image)
                window['-IMAGE2-'].update(data=get_img_bin_io(save_image_to_bytesio(image)).getvalue())

        elif event in (sg.WIN_CLOSED, '退出'):
            break

    # 关闭GUI窗口
    window.close()
