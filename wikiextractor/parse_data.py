import re
import json
import os


def parse_text3(text):
    result = []
    doc_pattern = re.compile(r'<doc id="(.*?)" url="(.*?)" title="(.*?)">(.*?)</doc>', re.DOTALL)
    matches = doc_pattern.findall(text)
    for match in matches:
        doc_id, url, title, content = match
        doc_dict = {
            "id": doc_id,
            "url": url,
            "title": title,
            "content": []
        }
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]  # 去除空行

        if len(lines) == 1 and lines[0] == title:  # 如果只有一行且与 title 相同，则舍弃
            continue

        if len(lines) > 0 and lines[0] == title:
            lines = lines[1:]

        main_title_content = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("##"):
                break
            main_title_content.append(lines[i])
            i += 1
        doc_dict["content"].append({
            "title": "",
            "content": main_title_content
        })

        while i < len(lines):
            if lines[i].startswith("##"):
                sub_title = lines[i].strip("#").strip()
                sub_title_dict = {
                    "title": sub_title,
                    "content": []
                }
                sub_title_content = []
                i += 1
                while i < len(lines):
                    if lines[i].startswith("###"):
                        sub_sub_title = lines[i].strip("#").strip()
                        sub_sub_title_dict = {
                            "title": sub_sub_title,
                            "content": []
                        }
                        sub_sub_title_content = []
                        i += 1
                        while i < len(lines):
                            if lines[i].startswith("####"):
                                sub_sub_sub_title = lines[i].strip("#").strip()
                                sub_sub_sub_title_dict = {
                                    "title": sub_sub_sub_title,
                                    "content": []
                                }
                                sub_sub_sub_title_content = []
                                i += 1
                                while i < len(lines):
                                    if lines[i].startswith("#####"):
                                        # 只把 ##### 删去，将内容添加到 sub_sub_sub_title_content 中
                                        sub_sub_sub_title_content.append(lines[i].replace("#####", ""))
                                    elif (lines[i].startswith("####") or lines[i].startswith("###") or lines[i].startswith("##")):
                                        break
                                    else:
                                        sub_sub_sub_title_content.append(lines[i])
                                    i += 1
                                sub_sub_sub_title_dict["content"] = sub_sub_sub_title_content
                                sub_sub_title_dict["content"].append(sub_sub_sub_title_dict)
                            elif lines[i].startswith("###") or lines[i].startswith("##"):
                                break
                            else:
                                sub_sub_title_content.append(lines[i])
                            i += 1
                        sub_sub_title_dict["content"].insert(0, {"title": "", "content": sub_sub_title_content})
                        sub_title_dict["content"].append(sub_sub_title_dict)
                    elif lines[i].startswith("##"):
                        break
                    else:
                        sub_title_content.append(lines[i])
                    i += 1
                sub_title_dict["content"].insert(0, {"title": "", "content": sub_title_content})
                doc_dict["content"].append(sub_title_dict)
            else:
                i += 1
        result.append(doc_dict)
    return result


def main1():
    input_file = './AA/wiki_11.txt'
    output_file = './AA_output5/1111111111.json'
    with open(input_file, 'r') as file:
        text = file.read()
    parsed_result = parse_text3(text)
    with open(output_file, 'w') as json_file:
        json.dump(parsed_result, json_file, indent=4)



def main2():
    input_dir = './en18_wiki'
    output_file = './output.json'
    error_log_file = 'error_log.json'
    processed_files_file = 'processed_files.txt'
    final_result = []
    error_log = []
    processed_files = []

    # 尝试读取已处理文件列表
    try:
        with open(processed_files_file, 'r') as f:
            processed_files = f.read().splitlines()
    except FileNotFoundError:
        pass

    # 尝试读取错误日志
    try:
        with open(error_log_file, 'r') as f:
            error_log = json.load(f)
    except FileNotFoundError:
        pass

    # 存储文件列表，用于判断文件是否被处理
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            sub_dir = os.path.join(root, dir)
            for sub_root, sub_dirs, sub_files in os.walk(sub_dir):
                for file in sub_files:
                    file_path = os.path.join(sub_root, file)
                    all_files.append(file_path)

    if not processed_files:
        processed_files = []

    try:
        for file_path in all_files:
            if file_path in processed_files:
                continue
            try:
                with open(file_path, 'r') as input_file:
                    text = input_file.read()
                parsed_result = parse_text3(text)
                final_result.extend(parsed_result)
                processed_files.append(file_path)
            except Exception as e:
                error_log.append({
                    "file_path": file_path,
                    "error": str(e)
                })
                continue
    except Exception as e:
        print(f"Error during traversal: {e}")
        # 保存错误日志
        with open(error_log_file, 'w') as f:
            json.dump(error_log, f, indent=4)
        return

    try:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_result = json.load(f)
            final_result = existing_result + final_result
        with open(output_file, 'w') as json_file:
            json.dump(final_result, json_file, indent=4)
    except Exception as e:
        print(f"Error writing to output file: {e}")
        # 保存错误日志
        with open(error_log_file, 'w') as f:
            json.dump(error_log, f, indent=4)
        return

    # 保存已处理文件列表
    with open(processed_files_file, 'w') as f:
        for file_path in processed_files:
            f.write(file_path + '\n')

    # 清除错误日志（如果成功完成）
    if not error_log:
        try:
            os.remove(error_log_file)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main2()
