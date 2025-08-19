#!/usr/bin/env python3
"""
Script principal para processamento de imagens
"""

import argparse
from src.image_processor import ImageProcessor
from src.utils import batch_process_images, validate_image_path

def main():
    parser = argparse.ArgumentParser(description='Processamento de Imagens - Conversão para Grayscale e Binária')
    parser.add_argument('--input', '-i', required=True, help='Caminho da imagem de entrada')
    parser.add_argument('--output', '-o', help='Caminho para salvar a imagem processada')
    parser.add_argument('--threshold', '-t', type=int, default=128, 
                       help='Threshold para binarização (0-255)')
    parser.add_argument('--batch', '-b', action='store_true', 
                       help='Processar todas as imagens de um diretório')
    parser.add_argument('--display', '-d', action='store_true', 
                       help='Exibir as imagens processadas')
    
    args = parser.parse_args()
    
    if args.batch:
        # Processamento em lote
        output_dir = args.output if args.output else './output_images'
        batch_process_images(args.input, output_dir)
    else:
        # Processamento de imagem única
        try:
            validate_image_path(args.input)
            
            processor = ImageProcessor(args.input)
            
            # Converte para grayscale
            if args.output:
                grayscale_path = args.output.replace('.', '_grayscale.')
                processor.to_grayscale(grayscale_path)
            else:
                processor.to_grayscale()
            
            # Converte para binária
            if args.output:
                binary_path = args.output.replace('.', '_binary.')
                processor.to_binary(threshold=args.threshold, save_path=binary_path)
            else:
                processor.to_binary(threshold=args.threshold)
            
            # Exibe estatísticas
            stats = processor.get_image_stats()
            print("\nEstatísticas da imagem:")
            for key, value in stats.items():
                print(f"{key}: {value}")
            
            # Exibe as imagens se solicitado
            if args.display:
                processor.display_images()
                
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    main()
