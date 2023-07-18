# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess

# 获取当前文件的绝对路径，并添加到系统路径中
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import cv2
import json
import numpy as np
import time
import logging
from copy import deepcopy

# 导入自定义的一些工具类
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results, draw_re_results
from tools.infer.predict_system import TextSystem
from ppstructure.layout.predict_layout import LayoutPredictor
from ppstructure.table.predict_table import TableSystem, to_excel
from ppstructure.utility import parse_args, draw_structure_result
from tools.infer.predict_system import aliyun_ocr_system
# 获取日志对象
logger = get_logger()
# 定义文本分析系统
class StructureSystem(object):
    #模型会根据参数进行初始化，包括图片方向预测器，版面预测器和文本系统。
    def __init__(self, args):

        # 初始化系统模式和恢复参数
        self.mode = args.mode
        self.recovery = args.recovery

        # 如果args中设置了image_orientation，使用paddleclas进行图片方向预测
        self.image_orientation_predictor = None
        if args.image_orientation:
            import paddleclas
            self.image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation")

        # 当系统模式为结构模式时，初始化相关的预测器
        if self.mode == 'structure':
            # 设置日志级别
            if not args.show_log:
                logger.setLevel(logging.INFO)
            # 当args.layout为false且args.ocr为true时，自动将args.ocr设为false
            if args.layout == False and args.ocr == True:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            args.drop_score = 0
            # 初始化模型
            self.layout_predictor = None
            self.text_system = None
            self.table_system = None
            # 如果args.layout为true，那么创建版面预测器和文本系统
            if args.layout:
                self.layout_predictor = LayoutPredictor(args)
                if args.ocr:
                    self.text_system = TextSystem(args)
            # 如果args.table为true，创建表格系统
            if args.table:
                if self.text_system is not None:
                    self.table_system = TableSystem(
                        args, self.text_system.text_detector,
                        self.text_system.text_recognizer)
                else:
                    self.table_system = TableSystem(args)

        # 当系统模式为'kie'时，初始化kie预测器
        elif self.mode == 'kie':
            from ppstructure.kie.predict_kie_token_ser_re import SerRePredictor
            self.kie_predictor = SerRePredictor(args)
    '''
        定义当对象被调用时的行为，对图片进行版面分析，并返回分析结果
            1.首先，方法创建了一个名为time_dict的字典，用于存储各个预测步骤所消耗的时间。
            
            2.如果系统初始化时设定了图像方向预测器（self.image_orientation_predictor），__call__方法会首先用这个预测器对图像进行处理。这个处理过程可能包括旋转图像，以使其具有正确的方向。对这个过程的耗时会记录在time_dict中。
            
            3.__call__方法接着检查系统的模式（self.mode）。如果系统模式为'structure'，方法会进行以下操作：
                3.1 如果版面预测器（self.layout_predictor）存在，使用它对图像进行预测，获取版面结果（layout_res）并记录预测耗时。
                3.2 接着，方法遍历版面预测结果中的每一个区域（region）。如果区域的label是'table'，并且系统中存在表格系统（self.table_system），那么就使用表格系统对这个区域进行预测。如果label不是'table'，并且存在文本系统（self.text_system），那么就使用文本系统对这个区域进行预测。
                3.3 在对区域进行预测的过程中，还会对预测结果进行一些处理，例如移除样式字符等。所有的预测结果都会被添加到res_list中。
                
            4. 如果系统模式为'kie'，__call__方法会使用kie预测器（self.kie_predictor）对图像进行预测，并记录预测耗时。
        
            5.最后，__call__方法会返回预测结果和time_dict。
    '''

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        # 初始化各类时间计数器
        time_dict = {
            'image_orientation': 0,
            'layout': 0,
            'table': 0,
            'table_match': 0,
            'det': 0,
            'rec': 0,
            'kie': 0,
            'all': 0
        }
        start = time.time()

        # 如果存在图像方向预测器，使用预测器对图像进行预处理，包括旋转等操作
        if self.image_orientation_predictor is not None:
            tic = time.time()
            cls_result = self.image_orientation_predictor.predict(
                input_data=img)
            cls_res = next(cls_result)
            angle = cls_res[0]['label_names'][0]
            cv_rotate_code = {
                '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv2.ROTATE_180,
                '270': cv2.ROTATE_90_CLOCKWISE
            }
            if angle in cv_rotate_code:
                img = cv2.rotate(img, cv_rotate_code[angle])
            toc = time.time()
            time_dict['image_orientation'] = toc - tic

        # 如果系统模式为结构模式，进行版面分析
        if self.mode == 'structure':
            ori_im = img.copy()
            if self.layout_predictor is not None:
                layout_res, elapse = self.layout_predictor(img)
                time_dict['layout'] += elapse
            else:
                h, w = ori_im.shape[:2]
                layout_res = [dict(bbox=None, label='table')]
            res_list = []
            for region in layout_res:
                res = ''

                # if region['label'] == 'header' or region['label'] == 'footer' or region['label'] == 'figure_caption' or region['label'] == 'table_caption':
                #     continue

                if region['bbox'] is not None:
                    x1, y1, x2, y2 = region['bbox']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                if region['label'] == 'table':
                    if self.table_system is not None:
                        res, table_time_dict = self.table_system(
                            roi_img, return_ocr_result_in_table)
                        time_dict['table'] += table_time_dict['table']
                        time_dict['table_match'] += table_time_dict['match']
                        time_dict['det'] += table_time_dict['det']
                        time_dict['rec'] += table_time_dict['rec']

                else:
                    if args.aliyun_ocr :
                        if self.recovery:
                            wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)
                            wht_im[y1:y2, x1:x2, :] = roi_img
                            ali_ocr_res_list, ocr_time_dict = aliyun_ocr_system(
                                wht_im)
                        else:
                            ali_ocr_res_list, ocr_time_dict = aliyun_ocr_system(
                                roi_img)
                        time_dict['det'] += ocr_time_dict['det']
                        time_dict['rec'] += ocr_time_dict['rec']

                        style_token = [
                            '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                            '</b>', '<sub>', '</sup>', '<overline>',
                            '</overline>', '<underline>', '</underline>', '<i>',
                            '</i>'
                        ]
                        for token in style_token:
                            res = [{'text':tmp_res.get('text').replace(token, ''), 'confidence': 0.99999, 'text_region': [[0, 0], [0, 0], [0, 0], [0, 0]]} for tmp_res in ali_ocr_res_list]
                    else:
                        if self.text_system is not None:
                            if self.recovery:
                                wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)
                                wht_im[y1:y2, x1:x2, :] = roi_img
                                filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                                    wht_im)
                            else:
                                filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                                    roi_img)
                            time_dict['det'] += ocr_time_dict['det']
                            time_dict['rec'] += ocr_time_dict['rec']

                            # 删除样式字符，当使用在 PubtabNet 数据集上训练的识别模型时，它会识别表中的文本格式，例如<b>
                            style_token = [
                                '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                                '</b>', '<sub>', '</sup>', '<overline>',
                                '</overline>', '<underline>', '</underline>', '<i>',
                                '</i>'
                            ]
                            res = []
                            for box, rec_res in zip(filter_boxes, filter_rec_res):
                                rec_str, rec_conf = rec_res
                                for token in style_token:
                                    if token in rec_str:
                                        rec_str = rec_str.replace(token, '')
                                if not self.recovery:
                                    box += [x1, y1]
                                res.append({
                                    'text': rec_str,
                                    'confidence': float(rec_conf),
                                    'text_region': box.tolist()
                                })
                res_list.append({
                    'type': region['label'].lower(),
                    'bbox': [x1, y1, x2, y2],
                    'img': roi_img,
                    'res': res,
                    'img_idx': img_idx
                })
            end = time.time()
            time_dict['all'] = end - start
            return res_list, time_dict

        # 如果系统模式为'kie'，进行'kie'预测
        elif self.mode == 'kie':
            re_res, elapse = self.kie_predictor(img)
            time_dict['kie'] = elapse
            time_dict['all'] = elapse
            return re_res[0], time_dict
        return None, None


'''
    结果保存函数，将版面分析的结果保存到文件中.
    如果区域的类型是表格，还会将预测结果转换为 Excel 格式并保存；如果区域的类型是图形，还会保存该区域的图片。
'''
def save_structure_res(res, save_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    # save res
    with open(
            os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
            'w',
            encoding='utf8') as f:
        #开始遍历res_cp中的每一个区域。
        for region in res_cp:
            #从区域中取出图片并删除该键值对。
            roi_img = region.pop('img')
            #将区域的信息转换为 JSON 格式，并写入到文本文件中。
            f.write('{}\n'.format(json.dumps(region)))

            #检查区域的类型是否为表格，并且是否包含有用的预测结果，并且预测结果中是否包含'html'字段。
            if region['type'].lower() == 'table' and len(region[
                    'res']) > 0 and 'html' in region['res']:
                excel_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.xlsx'.format(region['bbox'], img_idx))
                #将预测结果中的'html'字段转换为 Excel 格式并保存。
                to_excel(region['res']['html'], excel_path)
            #如果区域的类型是图形，那么执行以下操作。
            elif region['type'].lower() == 'figure':
                img_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.jpg'.format(region['bbox'], img_idx))
                #保存图形区域的图片。
                cv2.imwrite(img_path, roi_img)

# 主要包括读取图片，预测，保存结果，打印日志等步骤。
def main(args):
    #获取存放图片的文件夹中所有图片的文件路径。
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    #进行并行处理时使用的，它根据进程id和总进程数将图片文件列表分片，每个进程处理一个分片。
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    #是否使用pdf2docx API
    if not args.use_pdf2docx_api:
        #初始化一个StructureSystem对象。
        structure_sys = StructureSystem(args)
        #生成保存预测结果的文件夹路径。
        save_folder = os.path.join(args.output, structure_sys.mode)
        os.makedirs(save_folder, exist_ok=True)
    #获取图片文件列表的长度
    img_num = len(image_file_list)

    #遍历每一个图片文件
    for i, image_file in enumerate(image_file_list):
        #记录日志，输出正在处理的图片文件的序号和总数，以及图片文件的名字。
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        #读取图片文件，并检查图片是否是gif或者pdf的标志。
        img, flag_gif, flag_pdf = check_and_read(image_file)
        #获取图片文件的名字（不包括扩展名）。
        img_name = os.path.basename(image_file).split('.')[0]

        #如果已经设置了版面恢复选项，同时也使用了pdf2docx API，并且图片是PDF文件，则执行以下的代码。
        if args.recovery and args.use_pdf2docx_api and flag_pdf:
            from pdf2docx.converter import Converter
            os.makedirs(args.output, exist_ok=True)
            #定义了docx文件的路径和名字
            docx_file = os.path.join(args.output,
                                     '{}_api.docx'.format(img_name))

            #创建了一个Converter对象，用于将PDF文件转换为docx文件。
            cv = Converter(image_file)
            #将PDF文件转换为docx文件。
            cv.convert(docx_file)
            #关闭Converter对象。
            cv.close()
            logger.info('docx save to {}'.format(docx_file))
            continue

        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)

        if not flag_pdf:
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                continue
            # 如果图片读取成功，创建一个只包含这一张图片的列表。
            imgs = [img]
        else:
            # 将img赋值给imgs，因为check_and_read函数在处理PDF文件时返回的是图片列表。
            imgs = img

        #保存所有图片的处理结果。
        all_res = []
        for index, img in enumerate(imgs):
            res, time_dict = structure_sys(img, img_idx=index)
            img_save_path = os.path.join(save_folder, img_name,
                                         'show_{}.jpg'.format(index))
            os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
            #如果模式为'structure'并且处理结果不为空。
            if structure_sys.mode == 'structure' and res != [] :
                    #将处理结果绘制到原图片上。
                    draw_img = draw_structure_result(img, res, args.vis_font_path, args.aliyun_ocr)
                    save_structure_res(res, save_folder, img_name, index)
            #如果模式为'kie'
            elif structure_sys.mode == 'kie':
                #判断kie预测器是否存在
                if structure_sys.kie_predictor.predictor is not None:
                    draw_img = draw_re_results(
                        img, res, font_path=args.vis_font_path)
                else:
                    draw_img = draw_ser_results(
                        img, res, font_path=args.vis_font_path)

                with open(
                        os.path.join(save_folder, img_name, 'res_{}_kie.txt'.format(index)),'w',
                        encoding='utf8') as f:
                    #格式化处理结果为字符串。
                    res_str = '{}\t{}\n'.format(image_file,json.dumps(
                            {
                                "ocr_info": res
                            }, ensure_ascii=False))
                    f.write(res_str)
            if res != []:
                #将处理结果绘制后的图片保存。
                cv2.imwrite(img_save_path, draw_img)
                logger.info('result save to {}'.format(img_save_path))
            #如果设置了版面恢复选项，并且处理结果不为空。
            if args.recovery and res != []:
                from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
                #获取图片的高度和宽度。
                h, w, _ = img.shape
                #对处理结果按照布局进行排序。
                res = sorted_layout_boxes(res, w)
                #将处理结果添加到所有图片的处理结果列表中。
                all_res += res
        #如果设置了版面恢复选项，并且所有图片的处理结果不为空。
        if args.recovery and all_res != []:
            try:
                #尝试将处理结果恢复为docx文件。
                convert_info_docx(args, img, all_res, save_folder, img_name)
            except Exception as ex:
                #如果在恢复过程中出现异常 ,记录错误日志，输出错误信息。
                logger.error("error in layout recovery image:{}, err msg: {}".
                             format(image_file, ex))
                continue
        #记录日志，输出预测时间。
        logger.info("Predict time : {:.3f}s".format(time_dict['all']))


if __name__ == "__main__":
    args = parse_args()
    # 支持多进程处理，根据参数决定是否开启多进程
    if args.use_mp:
        # 创建多个子进程处理图片
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        # 单进程处理
        main(args)
