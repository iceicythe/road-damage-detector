"""Local batch inspection: python -m streamlit run app.py."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))


def main():
    import io
    import json
    import pandas as pd
    import streamlit as st
    # streamlit-drawable-canvas 0.9.x uses Streamlit's removed image_to_url helper.
    # Reuse Streamlit's media manager so the iframe receives a normal /media URL.
    import streamlit.elements.image as st_image
    if not hasattr(st_image, 'image_to_url'):
        from streamlit.elements.lib.image_utils import marshall_images
        from streamlit.elements.lib.layout_utils import create_layout_config
        from streamlit.proto.Image_pb2 import ImageList
        def image_to_media_url(image, width, clamp, channels, output_format, key):
            proto=ImageList()
            marshall_images(key, image, None, create_layout_config(width=width), proto,
                            clamp, channels, output_format)
            return proto.imgs[0].url
        st_image.image_to_url=image_to_media_url
    from PIL import Image
    from streamlit_drawable_canvas import st_canvas
    from road_damage.common import ROOT, local_path
    from road_damage.experiments import yolo_class
    from road_damage.inspection import (STORE,LABELS,create_batch,load_batch,process_batch,
        editor_rows,validate_edits,save_review,current_boxes,review_sort_key,annotated,export_batch)

    st.set_page_config(page_title='路面巡检 · 检测与复核',page_icon='🛣️',layout='wide')
    st.title('路面巡检 · 检测与复核')
    st.caption('批量查看病害候选，逐图校正，再导出巡检记录。')

    @st.cache_resource
    def model_for(path,mtime):
        return yolo_class()(path,task='detect')

    st.sidebar.header('巡检批次')
    existing=sorted(STORE.glob('*/batch.json'),reverse=True) if STORE.exists() else []
    choices={p.parent.name:p.parent for p in existing}
    def display_batch(ident):
        if ident=='新建批次': return ident
        b=load_batch(choices[ident]); return f"{b['title']} · {ident[:15]}"
    selection=st.sidebar.selectbox('打开已保存批次',['新建批次',*choices],format_func=display_batch,key='batch_select')
    if st.sidebar.button('打开批次',disabled=selection=='新建批次'):
        st.session_state.active_batch=str(choices[selection]); st.rerun()
    if st.sidebar.button('新建巡检'):
        st.session_state.pop('active_batch',None); st.rerun()

    if 'active_batch' not in st.session_state:
        st.subheader('开始一批巡检')
        title=st.text_input('批次名称',value='道路巡检')
        source=st.radio('图片来源',['上传图片','体验示例'],horizontal=True)
        uploads=st.file_uploader('选择道路照片（最多 100 张，单张不超过 20MB）',type=['jpg','jpeg','png'],accept_multiple_files=True,max_upload_size=20) if source=='上传图片' else []
        if source=='体验示例': st.info('使用 3 张现有验证集图片体验流程。示例用于展示，不作为新增评测成绩。')
        with st.expander('检测设置'):
            weights=st.text_input('模型文件',value='runs/train/n640/weights/best.pt')
            device=st.selectbox('运行设备',['0','cpu'],format_func=lambda x:'NVIDIA 显卡' if x=='0' else 'CPU')
            conf=st.slider('置信度阈值',.05,.95,.25,.05)
            st.caption('阈值越低，候选框通常越多，也需要更多复核。')
        if st.button('开始批量检测',type='primary',disabled=source=='上传图片' and not uploads):
            try:
                weight=local_path(weights)
                if not weight.is_file(): raise ValueError('模型文件不存在。')
                if source=='体验示例':
                    rows=[json.loads(x) for x in (ROOT/'data/processed/rdd_v1/splits/val.jsonl').read_text(encoding='utf-8').splitlines()]
                    selected=[next(r for r in rows if r['labels'] and any(b[0]==1 for b in r['labels'])),
                              next(r for r in rows if r['labels'] and any(b[0]==3 for b in r['labels'])),
                              next(r for r in rows if not r['labels'])]
                    inputs=[(Path(r['image']).name,(ROOT/'data/processed/rdd_v1'/r['prepared_image']).read_bytes()) for r in selected]
                else: inputs=[(f.name,f.getvalue()) for f in uploads]
                directory=create_batch(inputs,title,weight,device,conf)
                st.session_state.active_batch=str(directory)
                with st.spinner('正在加载模型并检测图片…'):
                    model=model_for(str(weight),weight.stat().st_mtime_ns); bar=st.progress(0)
                    process_batch(directory,model,lambda i,n:bar.progress(i/n,text=f'已处理 {i}/{n} 张'))
                st.rerun()
            except Exception as exc:
                st.error(f'检测未完成：{exc}')
                if 'active_batch' in st.session_state: st.info('批次已保存。可在左侧重新打开，重试未完成图片。')
        return

    directory=Path(st.session_state.active_batch); batch=load_batch(directory)
    st.subheader(batch['title'])
    done=sum(i['detection_status']=='complete' for i in batch['items'])
    reviewed=sum(i['review_status']=='reviewed' for i in batch['items'])
    total=sum(len(current_boxes(i)) for i in batch['items'])
    c1,c2,c3,c4=st.columns(4)
    c1.metric('图片',len(batch['items'])); c2.metric('已检测',done)
    c3.metric('已复核',reviewed); c4.metric('当前框数',total)
    st.caption('框数按图片累计；同一病害出现在多张图片时，会重复计数。')
    if done<len(batch['items']):
        st.warning('部分图片尚未完成检测，可以继续处理。')
        if st.button('重试未完成图片'):
            try:
                weight=Path(batch['model']['weights'])
                with st.spinner('正在继续检测…'):
                    process_batch(directory,model_for(str(weight),weight.stat().st_mtime_ns))
                st.rerun()
            except Exception as exc: st.error(str(exc))
    st.divider()
    filters=st.radio('图片范围',['全部','待复核','已复核','检测失败'],horizontal=True)
    items=[i for i in batch['items'] if filters=='全部' or
           (filters=='待复核' and i['review_status']=='pending') or
           (filters=='已复核' and i['review_status']=='reviewed') or
           (filters=='检测失败' and i['detection_status']=='error')]
    if items:
        sort_mode=st.selectbox('复核队列排序',['原始顺序','低置信度优先','候选框多优先','空图抽查'],
            help='排序只改变查看顺序，不修改批次或模型结果。低置信度和空图适合优先人工确认。')
        if sort_mode!='原始顺序': items=sorted(items,key=lambda x:review_sort_key(x,sort_mode))
        ident=st.selectbox('选择图片',[i['id'] for i in items],format_func=lambda x:next(
            f"{i['filename']} · {'已复核' if i['review_status']=='reviewed' else '待复核'} · {len(current_boxes(i))} 框" for i in items if i['id']==x),key=f'image_{batch["id"]}_{filters}')
        item=next(i for i in items if i['id']==ident)
        if item['detection_status']=='error': st.error(item['error'])
        elif item['detection_status']=='complete':
            key=f'{batch["id"]}_{ident}_{item["revision"]}'
            with Image.open(directory/item['image']) as im: original=im.convert('RGB')
            st.caption(f'图片尺寸 {item["width"]} × {item["height"]} 像素。编号用于对应画面中的框。')
            frame=pd.DataFrame(editor_rows(item),columns=['编号','类别','x1','y1','x2','y2'])
            frame=frame.astype({'编号':'string','类别':'string','x1':'float64','y1':'float64','x2':'float64','y2':'float64'})
            left,right=st.columns(2)
            left.image(annotated(original,item['original_predictions']),caption='模型原始预测')
            preview=right.empty(); grid=st.checkbox('显示坐标网格，辅助补框',key='grid_'+key)
            st.markdown('**修正检测框**')
            st.caption('双击表格单元格修改；选中一行后可删除误报。漏检框直接在下方图片上拖拽矩形，再选择类别并保存。')
            edited=st.data_editor(frame,num_rows='delete',hide_index=True,width='stretch',key='editor_'+key,
                disabled=['编号'],column_config={'编号':st.column_config.TextColumn('编号'),
                '类别':st.column_config.SelectboxColumn('病害类别',options=LABELS,required=True),
                **{k:st.column_config.NumberColumn(k,min_value=0,max_value=item['width'] if k.startswith('x') else item['height'],step=1,required=True) for k in ['x1','y1','x2','y2']}})
            edited_rows=edited.to_dict('records'); valid=True; boxes=[]; canvas_rows=[]
            try:
                boxes=validate_edits(edited_rows,item)
            except ValueError as exc:
                valid=False; st.warning(str(exc)); preview.image(annotated(original,current_boxes(item),grid=grid),caption='已保存结果')
            if valid:
                st.markdown('**在图上补充漏检框**')
                draw_category=st.selectbox('新框类别',LABELS,key='canvas_category_'+key)
                canvas_width=min(900,item['width'])
                canvas_height=max(1,round(item['height']*canvas_width/item['width']))
                canvas_background=annotated(original,boxes,grid=grid).resize((canvas_width,canvas_height))
                canvas_result=st_canvas(
                    fill_color='rgba(231, 76, 60, 0.18)', stroke_width=3,
                    stroke_color='#e74c3c', background_color='#ffffff',
                    background_image=canvas_background, update_streamlit=True,
                    height=canvas_height, width=canvas_width, drawing_mode='rect',
                    key='canvas_'+key)
                if canvas_result.json_data and canvas_result.json_data.get('objects'):
                    scale_x=item['width']/canvas_width; scale_y=item['height']/canvas_height
                    for obj in canvas_result.json_data['objects']:
                        left=float(obj.get('left',0)); top=float(obj.get('top',0))
                        width=float(obj.get('width',0))*float(obj.get('scaleX',1))
                        height=float(obj.get('height',0))*float(obj.get('scaleY',1))
                        if width > 2 and height > 2:
                            canvas_rows.append({'类别':draw_category,
                                'x1':left*scale_x,'y1':top*scale_y,
                                'x2':(left+width)*scale_x,'y2':(top+height)*scale_y})
                if canvas_rows:
                    try:
                        boxes=validate_edits([*edited_rows,*canvas_rows],item)
                    except ValueError as exc:
                        valid=False; st.warning(str(exc))
                preview.image(annotated(original,boxes,grid=grid),caption='当前编辑预览（保存后生效）')
                st.caption(f'已绘制 {len(canvas_rows)} 个待添加框；拖拽结束后选择类别。坐标会自动换算回原图尺寸。')
            note=st.text_input('复核备注（可选）',key='note_'+key)
            if st.button('保存复核结果',type='primary',disabled=not valid,key='save_'+key):
                try:
                    save_review(directory,ident,[*edited_rows,*canvas_rows],note,item['revision'])
                    st.session_state.save_notice='复核结果已保存，原始预测和历次修改均已保留。'; st.rerun()
                except ValueError as exc: st.error(str(exc))
            if st.session_state.get('save_notice'): st.success(st.session_state.pop('save_notice'))
            with st.expander('坐标表单补框（备用）'):
                with st.form('add_box_'+key):
                    category=st.selectbox('新增框类别',LABELS)
                    columns=st.columns(4)
                    coordinates={}
                    for col,axis in zip(columns,['x1','y1','x2','y2']):
                        limit=item['width'] if axis.startswith('x') else item['height']
                        coordinates[axis]=col.number_input('新增 '+axis,min_value=0,max_value=limit,
                            value=0 if axis.endswith('1') else min(100,limit),step=1)
                    st.caption('提交会一起保存上方表格的修改，并记录一个新的复核版本。')
                    if st.form_submit_button('添加框并保存复核',disabled=not valid):
                        try:
                            save_review(directory,ident,[*edited_rows,{'类别':category,**coordinates}],note,item['revision'])
                            st.session_state.save_notice='新增框及当前修改已保存。'; st.rerun()
                        except ValueError as exc: st.error(str(exc))
            with st.expander(f'复核历史（{item["revision"]} 个版本）'):
                for event in reversed(item['history']):
                    st.write(f"版本 {event['revision']} · {event['at']} · {len(event['boxes'])} 框")
                    st.write(event['note'] or '无备注')
    else: st.info('当前范围没有图片。')
    st.divider(); st.subheader('导出巡检记录')
    st.caption('导出已保存的结果，包含原图、模型与当前标注图、逐图汇总、复核历史和已复核图片的 YOLO 标注。未保存编辑不会导出。')
    if st.button('生成导出包'):
        with st.spinner('正在整理结果…'):
            st.session_state.export_data=export_batch(directory); st.session_state.export_batch=batch['id']
            st.session_state.export_stamp=(directory/'batch.json').stat().st_mtime_ns
    if (st.session_state.get('export_batch')==batch['id'] and
        st.session_state.get('export_stamp')==(directory/'batch.json').stat().st_mtime_ns):
        st.download_button('下载 ZIP 结果包',st.session_state.export_data,file_name=f'{batch["id"]}.zip',mime='application/zip',on_click='ignore')


if __name__=='__main__': main()
