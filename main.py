# art_generator.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_art(seed=None, model_type='gan', style_image_path=None, content_image_path=None):
    """Generate artistic images using pre-trained models
    
    Args:
        seed (int): Random seed for reproducibility
        model_type (str): 'gan' or 'style' - which model to use
        style_image_path (str): Path to style image (for style transfer)
        content_image_path (str): Path to content image (for style transfer)
    """
    # Set random seed if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    if model_type == 'gan':
        # Using a pre-trained Progressive GAN from TF Hub
        print("Loading GAN model...")
        model = hub.load('https://tfhub.dev/google/progan-128/1').signatures['default']
        
        # Generate random noise vector
        noise = tf.random.normal([1, 512])
        
        # Generate image
        print("Generating artwork...")
        artwork = model(noise)['default'][0]
        artwork = (artwork + 1) / 2  # Convert from [-1,1] to [0,1]
        
    elif model_type == 'style':
        # Using style transfer model
        if not style_image_path or not content_image_path:
            raise ValueError("For style transfer, you need to provide both style and content image paths")
            
        print("Loading style transfer model...")
        model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        
        # Load images
        def load_image(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = image[tf.newaxis, :]
            return image
            
        content_image = load_image(content_image_path)
        style_image = load_image(style_image_path)
        
        # Generate stylized image
        print("Generating stylized artwork...")
        artwork = model(content_image, style_image)[0]
    else:
        raise ValueError("model_type must be either 'gan' or 'style'")
    
    # Display the artwork
    plt.figure(figsize=(10, 10))
    plt.imshow(artwork)
    plt.axis('off')
    plt.show()
    
    # Save the artwork
    output_path = 'generated_artwork.png'
    tf.keras.preprocessing.image.save_img(output_path, artwork)
    print(f"Artwork saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate art using TensorFlow')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='gan', 
                       choices=['gan', 'style'], help='Model type to use')
    parser.add_argument('--style', type=str, help='Path to style image (for style transfer)')
    parser.add_argument('--content', type=str, help='Path to content image (for style transfer)')
    
    args = parser.parse_args()
    
    generate_art(
        seed=args.seed,
        model_type=args.model,
        style_image_path=args.style,
        content_image_path=args.content
    )