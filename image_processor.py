import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path=None):
        """
        Inicializa o processador de imagens
        """
        self.image = None
        self.grayscale = None
        self.binary = None
        
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """
        Carrega uma imagem do caminho especificado
        """
        try:
            self.image = Image.open(image_path)
            print(f"Imagem carregada: {self.image.size[0]}x{self.image.size[1]}")
        except Exception as e:
            print(f"Erro ao carregar imagem: {e}")
    
    def to_grayscale(self, save_path=None):
        """
        Converte a imagem para níveis de cinza (0-255)
        """
        if self.image is None:
            raise ValueError("Nenhuma imagem carregada")
        
        # Converte para numpy array
        img_array = np.array(self.image)
        
        # Converte para grayscale usando a fórmula padrão
        if len(img_array.shape) == 3:  # Imagem colorida
            # Fórmula: 0.299*R + 0.587*G + 0.114*B
            self.grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:  # Já está em grayscale
            self.grayscale = img_array
        
        if save_path:
            self.save_image(self.grayscale, save_path)
        
        return self.grayscale
    
    def to_binary(self, threshold=128, save_path=None):
        """
        Converte a imagem para binária (0 e 255) usando um threshold
        """
        if self.grayscale is None:
            self.to_grayscale()
        
        # Aplica threshold
        self.binary = np.where(self.grayscale > threshold, 255, 0).astype(np.uint8)
        
        if save_path:
            self.save_image(self.binary, save_path)
        
        return self.binary
    
    def adaptive_binary(self, block_size=11, constant=2, save_path=None):
        """
        Binarização adaptativa usando o método de Otsu ou média local
        """
        if self.grayscale is None:
            self.to_grayscale()
        
        # Implementação simples de binarização adaptativa
        from scipy import ndimage
        
        # Calcula a média local
        local_mean = ndimage.uniform_filter(self.grayscale.astype(float), size=block_size)
        
        # Aplica threshold adaptativo
        self.binary = np.where(self.grayscale > (local_mean - constant), 255, 0).astype(np.uint8)
        
        if save_path:
            self.save_image(self.binary, save_path)
        
        return self.binary
    
    def save_image(self, image_array, save_path):
        """
        Salva a imagem processada
        """
        try:
            img = Image.fromarray(image_array)
            img.save(save_path)
            print(f"Imagem salva em: {save_path}")
        except Exception as e:
            print(f"Erro ao salvar imagem: {e}")
    
    def display_images(self, figsize=(15, 5)):
        """
        Exibe as imagens original, grayscale e binária
        """
        if self.image is None:
            print("Nenhuma imagem para exibir")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Imagem original
        axes[0].imshow(np.array(self.image))
        axes[0].set_title('Imagem Original')
        axes[0].axis('off')
        
        # Imagem em grayscale
        if self.grayscale is not None:
            axes[1].imshow(self.grayscale, cmap='gray')
            axes[1].set_title('Níveis de Cinza')
        else:
            axes[1].text(0.5, 0.5, 'Execute to_grayscale() primeiro', 
                        horizontalalignment='center', verticalalignment='center')
        axes[1].axis('off')
        
        # Imagem binária
        if self.binary is not None:
            axes[2].imshow(self.binary, cmap='gray')
            axes[2].set_title('Imagem Binária')
        else:
            axes[2].text(0.5, 0.5, 'Execute to_binary() primeiro', 
                        horizontalalignment='center', verticalalignment='center')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_image_stats(self):
        """
        Retorna estatísticas das imagens processadas
        """
        stats = {}
        
        if self.image is not None:
            stats['original_size'] = self.image.size
            stats['original_mode'] = self.image.mode
        
        if self.grayscale is not None:
            stats['grayscale_min'] = np.min(self.grayscale)
            stats['grayscale_max'] = np.max(self.grayscale)
            stats['grayscale_mean'] = np.mean(self.grayscale)
        
        if self.binary is not None:
            stats['binary_white_pixels'] = np.sum(self.binary == 255)
            stats['binary_black_pixels'] = np.sum(self.binary == 0)
            stats['binary_white_percentage'] = (stats['binary_white_pixels'] / 
                                               self.binary.size * 100)
        
        return stats
