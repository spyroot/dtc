import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np


def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, color='black', color_text='black', thickness=2):
    """

    :param image:
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :param display_str:
    :param color:
    :param color_text:
    :param thickness:
    :return:
    """
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str, fill=color_text, font=font)
    return image


def draw_boxes(disp_image, boxes, labels=None):
    """
    :param disp_image:
    :param boxes:
    :param labels:
    :return:
    """
    # xyxy format
    num_boxes = boxes.shape[0]
    list_gt = range(num_boxes)
    for i in list_gt:
        disp_image = _draw_single_box(disp_image,
                                      boxes[i, 0],
                                      boxes[i, 1],
                                      boxes[i, 2],
                                      boxes[i, 3],
                                      display_str=None if labels is None else labels[i],
                                      color='Red')
    return disp_image


def make_image(tensor, rescale=1, rois=None, labels=None):
    """
    Convert a numpy representation of an image to Image protobuf.

    :param tensor:
    :param rescale:
    :param rois:
    :param labels:
    :return:
    """
    from PIL import Image
    height, width, channel = tensor.shape
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)
    image = Image.fromarray(tensor)
    if rois is not None:
        image = draw_boxes(image, rois, labels=labels)
    image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return height, width, channel, image_string


def save_figure_to_numpy(fig):
    """

    :param fig:
    :return:
    """
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_filter_bank(melfb):
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)
    # ax.set(ylabel='Mel filter', title='Mel filter bank')
    # fig.colorbar(img, ax=ax)

    fig, axs = plt.subplots(1, 1)
    axs.set_title("Filter bank")
    axs.imshow(melfb, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    fig.canvas.draw()
    save_figure_to_numpy(fig)
    plt.close()


def plot_alignment_to_numpy(alignment, file_name=None, info=None):
    """

    :param file_name:
    :param alignment:
    :param info:
    :return:
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = None
    if file_name:
        plt.savefig(file_name)
    else:
        data = save_figure_to_numpy(fig)
    plt.close()

    return data


def plot_spectrogram_to_numpy(spectrogram, plot_size=(12, 3), file_name=None):
    """

    :param spectrogram:
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = None

    if file_name:
        plt.savefig(file_name)
    else:
        data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    """

    :param gate_targets:
    :param gate_outputs:
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
