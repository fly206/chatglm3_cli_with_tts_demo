import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
import asyncio

# 加入edge-tts库 及运行函数
import sys
from io import TextIOWrapper
from typing import Any, TextIO, Union
from edge_tts import Communicate, SubMaker, list_voices

# 加入音频播放库
from pydub import AudioSegment
from pydub.playback import play


# edge-tts 文字转语音
async def _run_tts(text, write_media, write_subtitles) -> None:
    words_in_cue=1
    # os.system("edge-tts -f tmp.txt -v zh-CN-XiaoyiNeural --write-media tmp.mp3  --write-subtitles tmp.vtt")
    tts: Communicate = Communicate(
        text=text,
        voice="zh-CN-XiaoyiNeural",
    )
    subs: SubMaker = SubMaker()
    with open(write_media, "wb") if write_media else sys.stdout.buffer as audio_file:
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                subs.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    sub_file: Union[TextIOWrapper, TextIO] = (
        open(write_subtitles, "w", encoding="utf-8") if write_subtitles else sys.stderr
    )
    with sub_file:
        sub_file.write(subs.generate_subs(words_in_cue))
        play(AudioSegment.from_file(write_media, "mp3"))


tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("ZhipuAI/chatglm3-6b", trust_remote_code=True).float()
# model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
model = model.eval()

os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"
stop_stream = False

welcome_prompt = (
    "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
)


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


async def main() -> None:
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        # glm_response_file = open("tmp.txt", "w+")
        glm_response = ""
        for response, history, past_key_values in model.stream_chat(
            tokenizer,
            query,
            history=history,
            past_key_values=past_key_values,
            return_past_key_values=True,
        ):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                # glm_response_file.writelines(response[current_length:])
                glm_response += response[current_length:];
                current_length = len(response)
        print("")
        # glm_response_file.close()
        await _run_tts(glm_response, "tmp.mp3", "tmp.vtt")


if __name__ == "__main__":
    asyncio.run(main())
