# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import ast
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tools.infer.utility import draw_ocr_box_txt, str2bool, init_args as infer_args


def init_args():
    parser = infer_args()

    parser.add_argument(
        "--aliyun_ocr",
        type=bool,
        default=True,
        help='是否使用阿里云ocr接口')
    parser.add_argument(
        "--set_doc_chinese_font",
        type=str,
        default='宋体',
        help='设置文档全局中文字体')
    parser.add_argument(
        "--set_doc_english_font",
        type=str,
        default='Times New Roman',
        help='设置文档全局英文字体')
    parser.add_argument(
        "--set_doc_base_font_size",
        type=float,
        default=6.5,
        help='设置文档全局字体大小')
    parser.add_argument(
        "--set_doc_text_font_size",
        type=float,
        default=10,
        help='设置文档正文字体大小')
    parser.add_argument(
        "--set_doc_text_line_spacing",
        type=float,
        default=1.15,
        help='设置文档正文行间距')
    parser.add_argument(
        "--set_doc_text_space_after",
        type=float,
        default=5,
        help='设置文档段落后间距')
    parser.add_argument(
        "--set_doc_figure_and_table_caption_font_size",
        type=float,
        default=6.5,
        help='设置文档图片和表格标题字体大小')
    parser.add_argument(
        "--set_doc_table_font_size",
        type=float,
        default=10,
        help='设置文档表格字体大小')
    parser.add_argument(
        "--set_doc_table_style",
        type=str,
        default='TableGrid',
        help='设置表格风格')


    parser.add_argument("--output", type=str, default='./output')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default='TableAttn')
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument(
        "--merge_no_span_structure", type=str2bool, default=True)
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        default="../ppocr/utils/dict/table_structure_dict_ch.txt")
    # params for layout
    parser.add_argument("--layout_model_dir", type=str)
    parser.add_argument(
        "--layout_dict_path",
        type=str,
        default="../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt")
    parser.add_argument(
        "--layout_score_threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        "--layout_nms_threshold",
        type=float,
        default=0.5,
        help="Threshold of nms.")
    # params for kie
    parser.add_argument("--kie_algorithm", type=str, default='LayoutXLM')
    parser.add_argument("--ser_model_dir", type=str)
    parser.add_argument("--re_model_dir", type=str)
    parser.add_argument("--use_visual_backbone", type=str2bool, default=True)
    parser.add_argument(
        "--ser_dict_path",
        type=str,
        default="../train_data/XFUND/class_list_xfun.txt")
    # need to be None or tb-yx
    parser.add_argument("--ocr_order_method", type=str, default=None)
    # params for inference
    parser.add_argument(
        "--mode",
        type=str,
        choices=['structure', 'kie'],
        default='structure',
        help='structure and kie is supported')
    parser.add_argument(
        "--image_orientation",
        type=bool,
        default=False,
        help='Whether to enable image orientation recognition')
    parser.add_argument(
        "--layout",
        type=str2bool,
        default=True,
        help='Whether to enable layout analysis')
    parser.add_argument(
        "--table",
        type=str2bool,
        default=True,
        help='In the forward, whether the table area uses table recognition')
    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=True,
        help='In the forward, whether the non-table area is recognition by ocr')
    # param for recovery
    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=False,
        help='Whether to enable layout of recovery')
    parser.add_argument(
        "--use_pdf2docx_api",
        type=str2bool,
        default=False,
        help='Whether to use pdf2docx api')

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()

#该函数用于将版面分析处理结果绘制到原图片上
def draw_structure_result(image, result, font_path, use_aliyun_ocr=False):
    #检查输入的图片是否是numpy数组，如果是，则将其转换为PIL的Image对象
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    #初始化三个空列表，用于保存文本框的位置、文本内容以及文本的置信度
    boxes, txts, scores = [], [], []

    #复制原始图片，以在复制的图片上绘制处理结果
    img_layout = image.copy()
    #创建一个ImageDraw对象，用于在图片上绘制
    draw_layout = ImageDraw.Draw(img_layout)
    #定义文本的颜色
    text_color = (255, 255, 255)
    #定义文本背景的颜色
    text_background_color = (80, 127, 255)
    #初始化一个字典，用于保存每种类型的颜色
    catid2color = {}
    #定义字体的大小
    font_size = 15
    #创建一个字体对象
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    #遍历传入的处理结果，对每一个结果进行处理
    for region in result:
        #如果当前的类型还没有对应的颜色，则随机生成一个颜色
        if region['type'] not in catid2color and region['type'] != 'line':
            box_color = (random.randint(0, 255), random.randint(0, 255),
                         random.randint(0, 255))
            #将生成的颜色保存到字典中
            catid2color[region['type']] = box_color
        else:
            box_color = catid2color[region['type']]
        #获取当前结果的边界框位置
        box_layout = region['bbox']
        #在图片上绘制边界框
        draw_layout.rectangle(
            [(box_layout[0], box_layout[1]), (box_layout[2], box_layout[3])],
            outline=box_color,
            width=3)
        #获取当前类型文本的宽度和高度
        text_w, text_h = font.getsize(region['type'])
        #在图片上绘制文本背景
        draw_layout.rectangle(
            [(box_layout[0], box_layout[1]),
             (box_layout[0] + text_w, box_layout[1] + text_h)],
            fill=text_background_color)
        #在图片上绘制文本
        draw_layout.text(
            (box_layout[0], box_layout[1]),
            region['type'],
            fill=text_color,
            font=font)
        #如果当前结果的类型是表格,则跳过
        if region['type'] == 'table' or use_aliyun_ocr:
            pass
        else:
            #遍历当前结果的所有文本结果。
            for text_result in region['res']:
                #将文本结果的位置添加到boxes列表中。
                boxes.append(np.array(text_result['text_region']))
                #将文本结果的内容添加到txts列表中。
                txts.append(text_result['text'])
                #将文本结果的置信度添加到scores列表中。
                scores.append(text_result['confidence'])
    #调用draw_ocr_box_txt函数，将所有的文本结果绘制到图片上。
    im_show = draw_ocr_box_txt(
        img_layout, boxes, txts, scores, font_path=font_path, drop_score=0)
    return im_show
