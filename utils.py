import matplotlib.pyplot as plt
import torch


def show_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_image_attn(img, title=None):
    """Imshow for Tensor."""

    # unnormalize
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    plt.axis("off")
    plt.imshow(img)
    if title is not None:
        cleaned_text = title.replace("<EOS>", "").strip()
        plt.title(cleaned_text)
    plt.pause(0.001)


def get_caps_from(model, features_tensors, device, dataset):
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
        caption = " ".join(caps)
        show_image(features_tensors[0], title=caption)

    return caps, alphas


def plot_attention(img, result, attention_plot):
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7, 7)

        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        plt.axis("off")
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap="gray", alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    plt.axis("off")
    plt.show()
