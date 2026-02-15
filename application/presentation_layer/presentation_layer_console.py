from application.mid_layer.mid_layer import MidLayer
from application.presentation_layer.translator import translate_engine_output


def main():
    input_path = "../image_for_inference.jpg"
    mid_layer = MidLayer()
    output = mid_layer.decide_if_aurora_or_detect_other_phenomena(input_path)
    print(translate_engine_output(output))

if __name__ == '__main__':
    main()