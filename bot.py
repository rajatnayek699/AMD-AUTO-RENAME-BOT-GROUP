#!/usr/bin/env python3
"""
Auto Rename Bot - Enhanced Version with Admin Priority Interruption System
FINAL VERSION: Admin tasks interrupt normal tasks, with proper task management
"""

import os
import re
import sys
import time
import json
import math
import asyncio
import logging
import datetime
import shutil
import subprocess
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
from dotenv import load_dotenv
from PIL import Image
import motor.motor_asyncio
from pyrogram import Client, filters, __version__, idle
from pyrogram.types import (
    Message, InlineKeyboardButton, InlineKeyboardMarkup, 
    CallbackQuery
)

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
class Config:
    API_ID = int(os.getenv("API_ID", "25775944"))
    API_HASH = os.getenv("API_HASH", "217e861ebca9da0dd4c17b1abf92636c")
    BOT_TOKEN = os.getenv("BOT_TOKEN", "7750507797:AAFT5QgxqdKnqDBu_2ZkjFxo9u5fBNOF5qY")
    ADMIN = [int(admin) for admin in os.getenv("ADMIN", "1869817167").split(",") if admin.strip()]
    DB_URL = os.getenv("DB_URL", "mongodb+srv://Filex:Guddu8972771037@cluster0.er3kfsr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    DB_NAME = os.getenv("DB_NAME", "Filex")
    LOG_CHANNEL = int(os.getenv("LOG_CHANNEL", "-1002795055491"))
    START_PIC = os.getenv("START_PIC", "https://graph.org/file/29a3acbbab9de5f45a5fe.jpg")
    BOT_UPTIME = time.time()
    TIMEOUT_SECONDS = 300  # 5 minutes timeout for stuck tasks

class Txt:
    START_TXT = """<b>ʜᴇʏ! {}  

» ɪ ᴀᴍ ᴀᴅᴠᴀɴᴄᴇᴅ ʀᴇɴᴀᴍᴇ ʙᴏᴛ! ᴡɪᴛʜ ᴀᴅᴍɪɴ ᴘʀɪᴏʀɪᴛʏ ɪɴᴛᴇʀʀᴜᴘᴛɪᴏɴ sʏsᴛᴇᴍ</b>"""
    
    HELP_TXT = """<b>📚 Available Commands:</b>

<b>⚙️ Setup Commands:</b>
• /autorename [format] - Set auto rename format
• /set_caption [caption] - Set custom caption
• /settitle [title] - Set metadata title
• /setauthor [author] - Set metadata author
• /setartist [artist] - Set metadata artist
• /setaudio [audio] - Set audio metadata
• /setsubtitle [subtitle] - Set subtitle metadata
• /setvideo [video] - Set video metadata
• /thumbnail - Set thumbnail from replied photo

<b>📊 View Commands:</b>
• /view_caption - View your caption
• /view_thumb - View your thumbnail
• /showmetadata - Show metadata settings
• /queue - Check processing queue
• /view_autorename - View your current autorename format

<b>🗑️ Delete Commands:</b>
• /del_caption - Delete caption
• /del_thumb - Delete thumbnail
• /resetmetadata - Reset metadata to default

<b>⚡ Control Commands:</b>
• /metadata - Toggle metadata ON/OFF
• /mediatype - Set media type preference
• /cancel - Cancel your current task

<b>👑 Admin Commands:</b>
• /stats - Bot statistics
• /clearqueue - Clear processing queue (admin only)
• /restart - Restart bot (admin only)

<b>⚠️ Admin Priority System:</b>
• When admin sends a file, current task is paused
• Admin task is processed immediately
• After admin task, paused task resumes
• Other users get paused notifications

<b>📖 Guide:</b>
1. First use /autorename to set rename format
2. Send any file in group to auto rename
3. Use /queue to check processing status
4. Admins have priority interruption"""

# ==================== ENHANCED QUEUE SYSTEM WITH ADMIN INTERRUPTION ====================
class PriorityQueue:
    def __init__(self):
        self.queue = []  # Min-heap with (priority, timestamp, task_id, task_data)
        self.task_counter = 0
        self.current_task = None
        self.current_task_id = None
        self.is_processing = False
        self.paused_tasks = []  # Store paused tasks for admin priority
        self.admin_mode_active = False  # Track if admin mode is active
        self.lock = asyncio.Lock()
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_start_time = None
        self.timeout_seconds = Config.TIMEOUT_SECONDS
        self.user_tasks = {}  # Track tasks by user_id for cancellation
    
    def add_to_queue(self, message: Message, user_id: int):
        """Add a file to the processing queue"""
        task_id = f"{user_id}_{int(time.time())}_{self.task_counter}"
        
        # Determine priority (0 for admin, 1 for normal users)
        priority = 0 if user_id in Config.ADMIN else 1
        
        queue_item = {
            'task_id': task_id,
            'message_id': message.id,
            'chat_id': message.chat.id,
            'user_id': user_id,
            'file_name': '',
            'file_size': 0,
            'media_type': '',
            'added_time': time.time(),
            'status': 'waiting',
            'priority': priority,
            'is_admin': user_id in Config.ADMIN,
            'retry_count': 0
        }
        
        # Get file info
        if message.document:
            queue_item['file_name'] = message.document.file_name or "file"
            queue_item['file_size'] = message.document.file_size
            queue_item['media_type'] = 'document'
        elif message.video:
            queue_item['file_name'] = message.video.file_name or "video.mp4"
            queue_item['file_size'] = message.video.file_size
            queue_item['media_type'] = 'video'
        elif message.audio:
            queue_item['file_name'] = message.audio.file_name or "audio.mp3"
            queue_item['file_size'] = message.audio.file_size
            queue_item['media_type'] = 'audio'
        
        # Store task in user_tasks dict
        if user_id not in self.user_tasks:
            self.user_tasks[user_id] = []
        self.user_tasks[user_id].append(task_id)
        
        # Add to heap (priority, timestamp, counter, task)
        heapq.heappush(self.queue, (priority, time.time(), self.task_counter, queue_item))
        self.task_counter += 1
        
        return len(self.queue), task_id, queue_item
    
    def get_next_task(self):
        """Get the next task from queue"""
        if self.queue:
            priority, timestamp, counter, task = heapq.heappop(self.queue)
            return task
        return None
    
    def peek_next_task(self):
        """Peek at the next task without removing it"""
        if self.queue:
            priority, timestamp, counter, task = heapq.nsmallest(1, self.queue)[0]
            return task
        return None
    
    def pause_current_task(self):
        """Pause the current task for admin priority"""
        if self.current_task:
            self.current_task['status'] = 'paused'
            self.current_task['paused_time'] = time.time()
            self.paused_tasks.append(self.current_task)
            print(f"Task paused: {self.current_task['task_id']}")
            return self.current_task
        return None
    
    def resume_paused_tasks(self):
        """Resume paused tasks back to queue"""
        resumed_count = 0
        for task in self.paused_tasks:
            task['status'] = 'waiting'
            task['priority'] = 0  # Give high priority to resumed tasks
            heapq.heappush(self.queue, (0, time.time(), self.task_counter, task))
            self.task_counter += 1
            resumed_count += 1
            print(f"Task resumed: {task['task_id']}")
        
        self.paused_tasks.clear()
        return resumed_count
    
    def remove_task_by_id(self, task_id, user_id=None):
        """Remove a specific task from queue"""
        new_queue = []
        removed = False
        
        for priority, timestamp, counter, task in self.queue:
            if task['task_id'] != task_id:
                heapq.heappush(new_queue, (priority, timestamp, counter, task))
            else:
                removed = True
                # Also remove from user_tasks
                if user_id and user_id in self.user_tasks and task_id in self.user_tasks[user_id]:
                    self.user_tasks[user_id].remove(task_id)
        
        self.queue = new_queue
        heapq.heapify(self.queue)
        
        # Also check paused tasks
        for task in self.paused_tasks:
            if task['task_id'] == task_id:
                self.paused_tasks.remove(task)
                removed = True
                if user_id and user_id in self.user_tasks and task_id in self.user_tasks[user_id]:
                    self.user_tasks[user_id].remove(task_id)
        
        return removed
    
    def cancel_user_tasks(self, user_id):
        """Cancel all tasks for a specific user"""
        tasks_to_cancel = []
        
        # Check queue
        new_queue = []
        for priority, timestamp, counter, task in self.queue:
            if task['user_id'] == user_id:
                tasks_to_cancel.append(task['task_id'])
            else:
                heapq.heappush(new_queue, (priority, timestamp, counter, task))
        
        self.queue = new_queue
        heapq.heapify(self.queue)
        
        # Check paused tasks
        for task in self.paused_tasks[:]:
            if task['user_id'] == user_id:
                tasks_to_cancel.append(task['task_id'])
                self.paused_tasks.remove(task)
        
        # Remove from user_tasks
        if user_id in self.user_tasks:
            del self.user_tasks[user_id]
        
        return tasks_to_cancel
    
    def get_user_current_task(self, user_id):
        """Get current task for a user (if they own the currently processing task)"""
        if self.current_task and self.current_task['user_id'] == user_id:
            return self.current_task
        return None
    
    def get_queue_length(self):
        """Get current queue length"""
        return len(self.queue)
    
    def clear_queue(self, admin_only=False, user_id=None):
        """Clear the queue with optional filters"""
        if admin_only:
            # Clear only admin tasks
            new_queue = []
            for priority, timestamp, counter, task in self.queue:
                if not task.get('is_admin', False):
                    heapq.heappush(new_queue, (priority, timestamp, counter, task))
            self.queue = new_queue
            heapq.heapify(self.queue)
        elif user_id:
            # Clear tasks for specific user
            self.cancel_user_tasks(user_id)
        else:
            # Clear all
            self.queue.clear()
            self.user_tasks.clear()
    
    def get_queue_info(self):
        """Get detailed queue information"""
        info = {
            'total': len(self.queue),
            'current': self.current_task,
            'is_processing': self.is_processing,
            'completed': self.completed_tasks,
            'failed': self.failed_tasks,
            'paused': len(self.paused_tasks),
            'admin_mode': self.admin_mode_active,
            'waiting_list': [],
            'admin_waiting': 0,
            'user_waiting': 0
        }
        
        # Sort queue by priority and timestamp for display
        sorted_queue = sorted(self.queue, key=lambda x: (x[0], x[1]))
        
        for i, (priority, timestamp, counter, item) in enumerate(sorted_queue):
            info['waiting_list'].append({
                'position': i + 1,
                'task_id': item['task_id'],
                'file_name': item['file_name'][:50] if item['file_name'] else 'Unknown',
                'user_id': item['user_id'],
                'is_admin': item.get('is_admin', False),
                'priority': 'High' if priority == 0 else 'Normal',
                'waiting_time': time.time() - item['added_time']
            })
            
            if item.get('is_admin', False):
                info['admin_waiting'] += 1
            else:
                info['user_waiting'] += 1
        
        return info
    
    def check_timeout(self):
        """Check if current task has timed out"""
        if self.is_processing and self.task_start_time:
            elapsed = time.time() - self.task_start_time
            return elapsed > self.timeout_seconds
        return False

# Global queue instance
processing_queue = PriorityQueue()

# ==================== DATABASE ====================
class Database:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(Config.DB_URL)
        self.db = self.client[Config.DB_NAME]
        self.col = self.db.users
    
    def new_user(self, user_id):
        return {
            "_id": int(user_id),
            "join_date": datetime.now().isoformat(),
            "file_id": None,
            "caption": None,
            "metadata": True,
            "title": "Encoded by @Codeflix_Bots",
            "author": "@Codeflix_Bots",
            "artist": "@Codeflix_Bots",
            "audio": "By @Codeflix_Bots",
            "subtitle": "By @Codeflix_Bots",
            "video": "Encoded By @Codeflix_Bots",
            "format_template": None,
            "media_type": "document",
            "ban_status": {
                "is_banned": False,
                "ban_duration": 0,
                "banned_on": datetime.max.isoformat(),
                "ban_reason": ''
            }
        }
    
    async def add_user(self, user_id):
        try:
            user_id_int = int(user_id)
            user = await self.col.find_one({"_id": user_id_int})
            
            if not user:
                user_data = self.new_user(user_id_int)
                await self.col.insert_one(user_data)
                print(f"✅ Added new user: {user_id_int}")
            else:
                print(f"ℹ️ User already exists: {user_id_int}")
        except Exception as e:
            print(f"❌ Error adding user {user_id}: {e}")
            # Try upsert as fallback
            try:
                user_data = self.new_user(user_id_int)
                await self.col.update_one(
                    {"_id": user_id_int},
                    {"$setOnInsert": user_data},
                    upsert=True
                )
            except:
                pass
    
    async def is_user_exist(self, user_id):
        try:
            user = await self.col.find_one({"_id": int(user_id)})
            return bool(user)
        except:
            return False
    
    async def total_users_count(self):
        return await self.col.count_documents({})
    
    async def set_thumbnail(self, user_id, file_id):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"file_id": file_id}})
    
    async def get_thumbnail(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("file_id", None) if user else None
    
    async def set_caption(self, user_id, caption):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"caption": caption}})
    
    async def get_caption(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("caption", None) if user else None
    
    async def set_format_template(self, user_id, format_template):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"format_template": format_template}})
    
    async def get_format_template(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("format_template", None) if user else None
    
    async def set_media_preference(self, user_id, media_type):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"media_type": media_type}})
    
    async def get_media_preference(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("media_type", "document") if user else "document"
    
    async def get_metadata(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("metadata", True) if user else True
    
    async def set_metadata(self, user_id, metadata):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"metadata": metadata}})
    
    async def get_title(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("title", "Encoded by @Codeflix_Bots") if user else "Encoded by @Codeflix_Bots"
    
    async def set_title(self, user_id, title):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"title": title}})
    
    async def get_author(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("author", "@Codeflix_Bots") if user else "@Codeflix_Bots"
    
    async def set_author(self, user_id, author):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"author": author}})
    
    async def get_artist(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("artist", "@Codeflix_Bots") if user else "@Codeflix_Bots"
    
    async def set_artist(self, user_id, artist):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"artist": artist}})
    
    async def get_audio(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("audio", "By @Codeflix_Bots") if user else "By @Codeflix_Bots"
    
    async def set_audio(self, user_id, audio):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"audio": audio}})
    
    async def get_subtitle(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("subtitle", "By @Codeflix_Bots") if user else "By @Codeflix_Bots"
    
    async def set_subtitle(self, user_id, subtitle):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"subtitle": subtitle}})
    
    async def get_video(self, user_id):
        user = await self.col.find_one({"_id": int(user_id)})
        return user.get("video", "Encoded By @Codeflix_Bots") if user else "Encoded By @Codeflix_Bots"
    
    async def set_video(self, user_id, video):
        await self.col.update_one({"_id": int(user_id)}, {"$set": {"video": video}})

# Initialize database
db = Database()

# ==================== UTILITY FUNCTIONS ====================
def humanbytes(size):
    """Convert bytes to human readable format"""
    if not size:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def TimeFormatter(milliseconds: int) -> str:
    seconds, milliseconds = divmod(int(milliseconds), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    tmp = ((str(days) + "ᴅ, ") if days else "") + \
          ((str(hours) + "ʜ, ") if hours else "") + \
          ((str(minutes) + "ᴍ, ") if minutes else "") + \
          ((str(seconds) + "ꜱ, ") if seconds else "")
    return tmp[:-2] or "0 s"

async def progress_for_pyrogram(current, total, ud_type, message, start):
    now = time.time()
    diff = now - start
    if round(diff % 5.00) == 0 or current == total:
        percentage = current * 100 / total
        speed = current / diff
        elapsed_time = round(diff) * 1000
        time_to_completion = round((total - current) / speed) * 1000
        estimated_total_time = elapsed_time + time_to_completion

        elapsed_time = TimeFormatter(milliseconds=elapsed_time)
        estimated_total_time = TimeFormatter(milliseconds=estimated_total_time)

        progress = "{0}{1}".format(
            ''.join(["█" for _ in range(math.floor(percentage / 5))]),
            ''.join(["░" for _ in range(20 - math.floor(percentage / 5))])
        )
        
        tmp = f"""\n
<b>» Size</b> : {humanbytes(current)} | {humanbytes(total)}
<b>» Done</b> : {round(percentage, 2)}%
<b>» Speed</b> : {humanbytes(speed)}/s
<b>» ETA</b> : {estimated_total_time if estimated_total_time else "0 s"} """
        
        try:
            await message.edit(
                text=f"{ud_type}\n\n{progress}{tmp}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("• ᴄᴀɴᴄᴇʟ •", callback_data="close")]
                ])
            )
        except:
            pass

# ==================== FILE PROCESSING FUNCTIONS ====================
def extract_season_episode(filename):
    """Extract season and episode numbers from filename"""
    patterns = [
        (r'S(\d+)(?:E|EP)(\d+)', ('season', 'episode')),
        (r'S(\d+)[\s-]*(?:E|EP)(\d+)', ('season', 'episode')),
        (r'Season\s*(\d+)\s*Episode\s*(\d+)', ('season', 'episode')),
        (r'\[S(\d+)\]\[E(\d+)\]', ('season', 'episode')),
        (r'S(\d+)[^\d]*(\d+)', ('season', 'episode')),
        (r'(?:E|EP|Episode)\s*(\d+)', (None, 'episode')),
        (r'\b(\d+)\b', (None, 'episode'))
    ]
    
    for pattern, (season_group, episode_group) in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            season = match.group(1) if season_group else None
            episode = match.group(2) if episode_group else match.group(1)
            return season, episode
    return None, None

def extract_quality(filename):
    """Extract quality information from filename"""
    quality_patterns = [
        (r'\b(\d{3,4}[pi])\b', lambda m: m.group(1)),  # 1080p, 720p
        (r'\b(4k|2160p)\b', lambda m: "4K"),
        (r'\b(2k|1440p)\b', lambda m: "2K"),
        (r'\b(HDRip|HDTV|WEB-DL|WEBRip|BluRay)\b', lambda m: m.group(1)),
        (r'\[(\d{3,4}[pi])\]', lambda m: m.group(1))
    ]
    
    for pattern, extractor in quality_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return extractor(match)
    return "Unknown"

async def cleanup_files(*paths):
    """Safely remove files if they exist"""
    for path in paths:
        try:
            if path and os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
        except Exception as e:
            print(f"Error removing {path}: {e}")

async def process_thumbnail(thumb_path):
    """Process and resize thumbnail image"""
    if not thumb_path or not os.path.exists(thumb_path):
        return None
    
    try:
        with Image.open(thumb_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((320, 320))
            img.save(thumb_path, "JPEG", quality=85)
        return thumb_path
    except Exception as e:
        print(f"Thumbnail processing error: {e}")
        await cleanup_files(thumb_path)
        return None

async def add_metadata_correct(input_path, output_path, user_id):
    """Add metadata to media file"""
    # Find ffmpeg path
    ffmpeg_path = None
    for path in ['ffmpeg', '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/bin/ffmpeg']:
        if shutil.which(path):
            ffmpeg_path = path
            break
    
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg not found")
    
    # Get metadata from database
    title = await db.get_title(user_id)
    artist = await db.get_artist(user_id)
    author = await db.get_author(user_id)
    video_title = await db.get_video(user_id)
    audio_title = await db.get_audio(user_id)
    subtitle_title = await db.get_subtitle(user_id)
    
    # Clean metadata strings
    def clean_metadata(text):
        if not text:
            return ""
        text = str(text).replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.replace('"', '\\"').replace("'", "\\'")
        return text
    
    title = clean_metadata(title)
    artist = clean_metadata(artist)
    author = clean_metadata(author)
    video_title = clean_metadata(video_title)
    audio_title = clean_metadata(audio_title)
    subtitle_title = clean_metadata(subtitle_title)
    
    # Build ffmpeg command
    cmd = [
        ffmpeg_path,
        '-i', input_path,
        '-map', '0',
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-c:s', 'copy',
        '-metadata', f'title={title}',
        '-metadata', f'artist={artist}',
        '-metadata', f'author={author}',
        '-metadata:s:v', f'title={video_title}',
        '-metadata:s:a', f'title={audio_title}',
        '-metadata:s:s', f'title={subtitle_title}',
        '-y',
        output_path
    ]
    
    # Execute ffmpeg with timeout
    try:
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            ),
            timeout=300
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg error: {error_msg}")
            
    except asyncio.TimeoutError:
        raise RuntimeError("FFmpeg process timed out")
    
    if not os.path.exists(output_path):
        raise RuntimeError("Output file was not created")
    
    return output_path

# ==================== ENHANCED QUEUE WORKER WITH ADMIN INTERRUPTION ====================
async def queue_worker():
    """Worker that processes files from the queue with admin interruption"""
    while True:
        try:
            # Wait if queue is empty
            if processing_queue.get_queue_length() == 0 and len(processing_queue.paused_tasks) == 0:
                await asyncio.sleep(2)
                continue
            
            # Check if already processing
            if processing_queue.is_processing:
                # Check for timeout on current task
                if processing_queue.check_timeout():
                    print(f"⚠️ Task timeout detected: {processing_queue.current_task_id}")
                    
                    if processing_queue.current_task:
                        try:
                            await app.send_message(
                                chat_id=processing_queue.current_task['chat_id'],
                                text=f"⏰ **Task Timeout**\n\n"
                                     f"The processing has timed out after 5 minutes. Moving to next task."
                            )
                        except:
                            pass
                    
                    processing_queue.is_processing = False
                    processing_queue.current_task = None
                    processing_queue.current_task_id = None
                    processing_queue.failed_tasks += 1
                
                await asyncio.sleep(1)
                continue
            
            # Start processing next task
            async with processing_queue.lock:
                processing_queue.is_processing = True
                task = processing_queue.get_next_task()
                
                if not task:
                    processing_queue.is_processing = False
                    await asyncio.sleep(2)
                    continue
                
                # Update task status
                processing_queue.current_task = task
                processing_queue.current_task_id = task['task_id']
                processing_queue.task_start_time = time.time()
                task['status'] = 'processing'
                task['start_time'] = time.time()
                
                print(f"Processing task: {task['task_id']} - Admin: {task.get('is_admin')}")
                
                # If this is an admin task, activate admin mode
                if task.get('is_admin'):
                    processing_queue.admin_mode_active = True
                    try:
                        await app.send_message(
                            chat_id=task['chat_id'],
                            text="🚨 **ADMIN MODE ACTIVATED**\n\n"
                                 "Admin task is being processed with high priority."
                        )
                    except Exception as e:
                        print(f"Failed to notify admin mode: {e}")
                
                # Try to get the message from Telegram
                try:
                    message = await app.get_messages(
                        chat_id=task['chat_id'],
                        message_ids=task['message_id']
                    )
                    
                    if not message:
                        print(f"Message not found: {task['message_id']}")
                        processing_queue.completed_tasks += 1
                        processing_queue.failed_tasks += 1
                        processing_queue.is_processing = False
                        processing_queue.current_task = None
                        processing_queue.current_task_id = None
                        continue
                    
                    # Process the file with timeout protection
                    try:
                        await asyncio.wait_for(
                            process_queue_file(message, task['user_id'], task),
                            timeout=Config.TIMEOUT_SECONDS
                        )
                        processing_queue.completed_tasks += 1
                        print(f"✅ Task completed successfully: {task['task_id']}")
                        
                    except asyncio.TimeoutError:
                        print(f"⏰ Task {task['task_id']} timed out")
                        try:
                            await app.send_message(
                                chat_id=task['chat_id'],
                                text=f"⏰ **Task Timeout**\n\n"
                                     f"The processing has timed out after 5 minutes."
                            )
                        except:
                            pass
                        processing_queue.failed_tasks += 1
                    except Exception as e:
                        print(f"❌ Error in task processing: {str(e)[:100]}")
                        processing_queue.failed_tasks += 1
                    
                    # If this was an admin task, deactivate admin mode and resume paused tasks
                    if task.get('is_admin'):
                        processing_queue.admin_mode_active = False
                        
                        # Resume paused tasks
                        if processing_queue.paused_tasks:
                            resumed_count = processing_queue.resume_paused_tasks()
                            if resumed_count > 0:
                                print(f"Resumed {resumed_count} paused tasks after admin task")
                                # Notify each paused task's chat
                                for paused_task in processing_queue.paused_tasks:
                                    try:
                                        await app.send_message(
                                            chat_id=paused_task['chat_id'],
                                            text="🔄 **Task Resumed**\n\n"
                                                 f"Your file `{paused_task.get('file_name', 'Unknown')[:50]}` has been resumed.\n"
                                                 "Processing will continue shortly."
                                        )
                                    except:
                                        pass
                        
                        try:
                            await app.send_message(
                                chat_id=task['chat_id'],
                                text="✅ **ADMIN MODE DEACTIVATED**\n\n"
                                     "Admin task completed. Normal processing resumes."
                            )
                        except Exception as e:
                            print(f"Failed to notify admin mode deactivation: {e}")
                
                except Exception as e:
                    print(f"❌ Error processing task {task['task_id']}: {e}")
                    processing_queue.failed_tasks += 1
                
                finally:
                    # Clean up user tasks tracking
                    user_id = task['user_id']
                    if user_id in processing_queue.user_tasks and task['task_id'] in processing_queue.user_tasks[user_id]:
                        processing_queue.user_tasks[user_id].remove(task['task_id'])
                    
                    # Clear current task
                    processing_queue.is_processing = False
                    processing_queue.current_task = None
                    processing_queue.current_task_id = None
                    processing_queue.task_start_time = None
                    await asyncio.sleep(1)
        
        except Exception as e:
            print(f"Queue worker error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)

async def process_queue_file(message, user_id, task_info):
    """Process a single file from the queue"""
    # Check if user has set rename format
    format_template = await db.get_format_template(user_id)
    if not format_template:
        try:
            await app.send_message(
                chat_id=message.chat.id,
                text="❌ **Please set a rename format first!**\n\n"
                     "Use: `/autorename Your Format Here`",
                reply_to_message_id=message.id
            )
        except:
            pass
        return False
    
    # Get file info
    if message.document:
        file_id = message.document.file_id
        file_name = message.document.file_name or "file"
        file_size = message.document.file_size
        media_type = "document"
        duration = 0
    elif message.video:
        file_id = message.video.file_id
        file_name = message.video.file_name or "video.mp4"
        file_size = message.video.file_size
        media_type = "video"
        duration = message.video.duration
    elif message.audio:
        file_id = message.audio.file_id
        file_name = message.audio.file_name or "audio.mp3"
        file_size = message.audio.file_size
        media_type = "audio"
        duration = message.audio.duration
    else:
        return False
    
    # Extract filename components
    base_name = os.path.splitext(file_name)[0]
    ext = os.path.splitext(file_name)[1] or ('.mp4' if media_type == 'video' else '.mp3')
    
    season, episode = extract_season_episode(base_name)
    quality = extract_quality(base_name)
    
    # Replace variables in template
    new_filename = format_template
    replacements = {
        '{filename}': base_name,
        '{season}': season or '01',
        '{episode}': episode or '01',
        '{quality}': quality,
        '{filesize}': humanbytes(file_size),
        '{duration}': str(timedelta(seconds=duration)) if duration else '00:00:00',
    }
    
    for key, value in replacements.items():
        new_filename = new_filename.replace(key, value)
    
    # Clean filename
    new_filename = re.sub(r'[<>:"/\\|?*]', '', new_filename)
    new_filename = new_filename.strip() + ext
    
    # Send processing started message
    priority_text = "🚨 **ADMIN PRIORITY**\n" if task_info.get('is_admin') else ""
    status_msg = await app.send_message(
        chat_id=message.chat.id,
        text=f"{priority_text}🔄 **Processing Started**\n"
             f"**File:** `{file_name[:50]}`\n"
             f"**New Name:** `{new_filename[:50]}`",
        reply_to_message_id=message.id
    )
    
    download_path = f"downloads/{user_id}_{int(time.time())}{ext}"
    
    try:
        # Download with progress
        start_time = time.time()
        await status_msg.edit_text(f"📥 **Downloading...**\n`{file_name[:50]}`")
        
        # Download file
        try:
            file_path = await message.download(
                file_name=download_path,
                progress=progress_for_pyrogram,
                progress_args=("📥 Downloading...", status_msg, start_time)
            )
        except TypeError:
            file_path = await message.download(file_name=download_path)
        
        if not file_path or not os.path.exists(file_path):
            await status_msg.edit_text("❌ Download failed!")
            return False
        
        file_size = os.path.getsize(file_path)
        await status_msg.edit_text(f"✅ **Downloaded!**\n\n**File:** `{file_name[:50]}`\n**Size:** {humanbytes(file_size)}\n\n⚙️ **Processing file...**")
        
        # Process metadata if enabled
        output_path = file_path
        metadata_enabled = await db.get_metadata(user_id)
        
        if metadata_enabled:
            try:
                metadata_path = f"temp/{user_id}_metadata{ext}"
                await status_msg.edit_text("🔧 **Adding metadata...**")
                output_path = await add_metadata_correct(file_path, metadata_path, user_id)
                await cleanup_files(file_path)
                await status_msg.edit_text(f"✅ **Metadata added successfully!**")
            except Exception as e:
                print(f"Metadata error: {e}")
                await status_msg.edit_text(f"⚠️ Metadata skipped: {str(e)[:100]}")
                output_path = file_path
        else:
            await status_msg.edit_text("ℹ️ **Metadata disabled, skipping...**")
        
        # Get thumbnail
        thumb_path = None
        user_thumb = await db.get_thumbnail(user_id)
        
        if user_thumb:
            try:
                thumb_path = f"temp/{user_id}_thumb.jpg"
                await app.download_media(user_thumb, file_name=thumb_path)
                thumb_path = await process_thumbnail(thumb_path)
            except Exception as e:
                print(f"Thumbnail error: {e}")
                thumb_path = None
        
        # Get caption
        caption_template = await db.get_caption(user_id) or "{filename}"
        caption = caption_template.replace("{filename}", os.path.splitext(new_filename)[0])\
                                 .replace("{filesize}", humanbytes(file_size))\
                                 .replace("{duration}", str(timedelta(seconds=duration)) if duration else '00:00:00')
        
        # Get media type preference
        media_pref = await db.get_media_preference(user_id)
        
        await status_msg.edit_text("📤 **Uploading renamed file...**")
        
        # Upload file based on media preference
        upload_start = time.time()
        
        try:
            if media_pref == "document" or media_type == "document":
                await app.send_document(
                    chat_id=message.chat.id,
                    document=output_path,
                    caption=caption[:1024] if caption else None,
                    thumb=thumb_path,
                    file_name=new_filename,
                    progress=progress_for_pyrogram,
                    progress_args=("📤 Uploading...", status_msg, upload_start),
                    reply_to_message_id=message.id
                )
            elif media_pref == "video" and media_type == "video":
                await app.send_video(
                    chat_id=message.chat.id,
                    video=output_path,
                    caption=caption[:1024] if caption else None,
                    thumb=thumb_path,
                    duration=duration,
                    progress=progress_for_pyrogram,
                    progress_args=("📤 Uploading...", status_msg, upload_start),
                    reply_to_message_id=message.id
                )
            elif media_pref == "audio" and media_type == "audio":
                await app.send_audio(
                    chat_id=message.chat.id,
                    audio=output_path,
                    caption=caption[:1024] if caption else None,
                    thumb=thumb_path,
                    duration=duration,
                    progress=progress_for_pyrogram,
                    progress_args=("📤 Uploading...", status_msg, upload_start),
                    reply_to_message_id=message.id
                )
            else:
                # Fallback to document
                await app.send_document(
                    chat_id=message.chat.id,
                    document=output_path,
                    caption=caption[:1024] if caption else None,
                    thumb=thumb_path,
                    file_name=new_filename,
                    progress=progress_for_pyrogram,
                    progress_args=("📤 Uploading...", status_msg, upload_start),
                    reply_to_message_id=message.id
                )
            
            await status_msg.delete()
            await app.send_message(
                chat_id=message.chat.id,
                text=f"✅ **File renamed successfully!**\n**New name:** `{new_filename[:50]}`",
                reply_to_message_id=message.id
            )
            return True
            
        except Exception as upload_error:
            raise upload_error
            
    except Exception as e:
        error_text = f"❌ **Error:** {str(e)[:200]}"
        try:
            await status_msg.edit_text(error_text)
        except:
            try:
                await app.send_message(
                    chat_id=message.chat.id,
                    text=error_text,
                    reply_to_message_id=message.id
                )
            except:
                pass
        print(f"Error: {e}")
        return False
    finally:
        # Cleanup
        try:
            await cleanup_files(
                download_path if 'download_path' in locals() else None,
                output_path if 'output_path' in locals() and output_path != download_path else None,
                thumb_path if 'thumb_path' in locals() else None
            )
        except:
            pass

# ==================== BOT CLIENT ====================
# Create necessary directories
os.makedirs("downloads", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Initialize bot
app = Client(
    "auto_rename_bot",
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    bot_token=Config.BOT_TOKEN,
    workers=100,
    sleep_threshold=10,
)

# ==================== USER MANAGEMENT ====================
async def ensure_user(user_id):
    """Ensure user exists in database"""
    try:
        await db.add_user(user_id)
    except:
        pass

# ==================== COMMON HANDLERS ====================
@app.on_message(filters.command("start"))
async def start_handler(client, message):
    user = message.from_user
    await ensure_user(user.id)
    
    buttons = InlineKeyboardMarkup([
        [InlineKeyboardButton("📚 ʜᴇʟᴘ", callback_data='help'), InlineKeyboardButton("⚙️ ᴍᴇᴛᴀᴅᴀᴛᴀ", callback_data='metadata')],
        [
            InlineKeyboardButton('📢 ᴜᴘᴅᴀᴛᴇs', url='https://t.me/Codeflix_Bots'),
            InlineKeyboardButton('🆘 sᴜᴘᴘᴏʀᴛ', url='https://t.me/CodeflixSupport')
        ],
        [
            InlineKeyboardButton('📊 Queue Status', callback_data='queue_status'),
            InlineKeyboardButton('❌ ᴄʟᴏsᴇ', callback_data='close')
        ]
    ])
    
    try:
        await message.reply_photo(
            Config.START_PIC,
            caption=Txt.START_TXT.format(user.mention),
            reply_markup=buttons
        )
    except:
        await message.reply_text(
            Txt.START_TXT.format(user.mention),
            reply_markup=buttons
        )

@app.on_message(filters.command(["help", "h"]))
async def help_handler(client, message):
    await ensure_user(message.from_user.id)
    
    buttons = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 ʜᴏᴍᴇ", callback_data='home')],
        [InlineKeyboardButton("⚙️ ᴍᴇᴛᴀᴅᴀᴛᴀ", callback_data='metadata'), InlineKeyboardButton("📊 Queue", callback_data='queue_status')]
    ])
    
    await message.reply_text(
        Txt.HELP_TXT,
        reply_markup=buttons,
        disable_web_page_preview=True
    )

@app.on_message(filters.command("autorename"))
async def autorename_handler(client, message):
    user_id = message.from_user.id
    await ensure_user(user_id)
    
    if len(message.command) < 2:
        current_format = await db.get_format_template(user_id)
        if current_format:
            await message.reply_text(
                f"**📋 Your Current Autorename Format:**\n\n"
                f"`{current_format}`\n\n"
                "To change it, use: `/autorename New Format Here`\n\n"
                "**Available variables:**\n"
                "- `{filename}`: Original filename\n"
                "- `{season}`: Season number\n"
                "- `{episode}`: Episode number\n"
                "- `{quality}`: Video quality\n"
                "- `{filesize}`: File size\n"
                "- `{duration}`: Duration\n\n"
                "**Example:** `/autorename {filename} [S{season}E{episode}] - {quality}`"
            )
        else:
            await message.reply_text(
                "**Please provide a rename format!**\n\n"
                "**Example:** `/autorename {filename} [S{season}E{episode}] - {quality}`\n\n"
                "**Available variables:**\n"
                "- `{filename}`: Original filename\n"
                "- `{season}`: Season number\n"
                "- `{episode}`: Episode number\n"
                "- `{quality}`: Video quality\n"
                "- `{filesize}`: File size\n"
                "- `{duration}`: Duration (for videos)\n\n"
                "**Note:** This setting is saved per user and works in both private and groups."
            )
        return
    
    format_template = message.text.split(" ", 1)[1]
    await db.set_format_template(user_id, format_template)
    
    await message.reply_text(
        f"**✅ Rename format set successfully!**\n\n"
        f"**Your format:** `{format_template}`\n\n"
        "Now send me any file in this chat or any group (where I'm added) to rename it automatically."
    )

@app.on_message(filters.command("cancel"))
async def cancel_handler(client, message):
    user_id = message.from_user.id
    
    # Check if user has any current task
    current_task = processing_queue.get_user_current_task(user_id)
    if current_task:
        # User owns the current task - they can cancel it
        if processing_queue.current_task and processing_queue.current_task['user_id'] == user_id:
            processing_queue.is_processing = False
            processing_queue.current_task = None
            processing_queue.current_task_id = None
            processing_queue.task_start_time = None
            await message.reply_text("✅ **Your current task has been cancelled!**")
            return
    
    # Cancel all tasks for this user in queue
    cancelled_tasks = processing_queue.cancel_user_tasks(user_id)
    
    if cancelled_tasks:
        await message.reply_text(f"✅ **Cancelled {len(cancelled_tasks)} of your tasks from the queue!**")
    else:
        await message.reply_text("❌ **You don't have any tasks in the queue to cancel.**")

@app.on_message(filters.command("queue"))
async def queue_status_handler(client, message):
    await ensure_user(message.from_user.id)
    queue_info = processing_queue.get_queue_info()
    
    if queue_info['total'] == 0 and not queue_info['is_processing'] and queue_info['paused'] == 0:
        await message.reply_text("📭 **Queue is empty!**\nNo files in processing queue.")
        return
    
    status_text = "📊 **Queue Status**\n\n"
    
    # Admin mode status
    if queue_info['admin_mode']:
        status_text += "**Admin Mode:** 🚨 **ACTIVE**\n\n"
    else:
        status_text += "**Admin Mode:** ✅ **INACTIVE**\n\n"
    
    if queue_info['is_processing'] and queue_info.get('current_task'):
        current = queue_info['current_task']
        priority_text = "🚨 **ADMIN**" if current.get('is_admin') else "👤 **USER**"
        status_text += f"🔄 **Currently Processing ({priority_text}):**\n"
        status_text += f"   • `{current.get('file_name', 'Unknown')[:30]}`\n"
        status_text += f"   • User ID: `{current.get('user_id', 'Unknown')}`\n"
        if 'start_time' in current:
            elapsed = time.time() - current['start_time']
            status_text += f"   • Processing for: `{TimeFormatter(elapsed*1000)}`\n"
        status_text += "\n"
    
    # Show paused tasks
    if queue_info['paused'] > 0:
        status_text += f"⏸️ **Paused Tasks:** `{queue_info['paused']}`\n\n"
    
    status_text += f"📋 **Waiting in Queue:** `{queue_info['total']}` files\n"
    
    if queue_info['waiting_list']:
        status_text += "\n**Next 5 in Queue:**\n"
        for i, item in enumerate(queue_info['waiting_list'][:5]):
            priority_icon = "👑" if item['is_admin'] else "👤"
            wait_time = TimeFormatter(item['waiting_time'] * 1000)
            status_text += f"`{item['position']}.` {priority_icon} `{item['file_name'][:30]}...` (User: `{item['user_id']}`) - Waiting: `{wait_time}`\n"
        
        if len(queue_info['waiting_list']) > 5:
            status_text += f"\n... and `{len(queue_info['waiting_list']) - 5}` more files\n"
    
    status_text += f"\n**Statistics:**\n"
    status_text += f"• ✅ Completed: `{queue_info['completed']}`\n"
    status_text += f"• ❌ Failed: `{queue_info['failed']}`\n"
    status_text += f"• ⏸️ Paused: `{queue_info['paused']}`\n"
    status_text += f"• 📊 Total in System: `{queue_info['total'] + (1 if queue_info['is_processing'] else 0) + queue_info['paused']}`"
    
    buttons = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_queue")],
        [InlineKeyboardButton("🗑️ Cancel My Tasks", callback_data="cancel_my_tasks")]
    ])
    
    await message.reply_text(status_text, reply_markup=buttons)

# ==================== GROUP FILE HANDLER WITH ADMIN INTERRUPTION ====================
@app.on_message(filters.group & (filters.document | filters.video | filters.audio))
async def group_file_handler(client, message):
    try:
        user_id = message.from_user.id
        chat_id = message.chat.id
        
        print(f"Received file in group {chat_id} from user {user_id}")
        
        await ensure_user(user_id)
        
        # Check if user is admin
        is_admin = user_id in Config.ADMIN
        
        # Check if admin mode is active and user is not admin
        if processing_queue.admin_mode_active and not is_admin:
            await client.send_message(
                chat_id=chat_id,
                text="⛔ **Admin mode is active!**\n\n"
                     "Please wait until the current admin task is completed.\n"
                     "Your file has been added to queue and will be processed after admin task.",
                reply_to_message_id=message.id
            )
        
        # Check if user has autorename format set
        format_template = await db.get_format_template(user_id)
        
        if not format_template:
            await client.send_message(
                chat_id=chat_id,
                text=f"❌ **Please set an autorename format first!**\n\n"
                     f"Use: `/autorename Your Format Here`\n\n"
                     f"**Example:** `/autorename {{filename}} [S{{season}}E{{episode}}]`",
                reply_to_message_id=message.id
            )
            return
        
        # Add file to queue
        queue_position, task_id, queue_item = processing_queue.add_to_queue(message, user_id)
        
        # Get file name
        if message.document:
            file_name = message.document.file_name or "file"
            file_size = message.document.file_size
        elif message.video:
            file_name = message.video.file_name or "video.mp4"
            file_size = message.video.file_size
        elif message.audio:
            file_name = message.audio.file_name or "audio.mp3"
            file_size = message.audio.file_size
        else:
            file_name = "Unknown"
            file_size = 0
        
        # If admin is sending file and there's a current task, pause it
        if is_admin and processing_queue.current_task and not processing_queue.current_task.get('is_admin'):
            paused_task = processing_queue.pause_current_task()
            if paused_task:
                # Notify the user whose task was paused
                try:
                    await client.send_message(
                        chat_id=paused_task['chat_id'],
                        text="⏸️ **Task Paused**\n\n"
                             f"Your file `{paused_task.get('file_name', 'Unknown')[:50]}` has been paused.\n"
                             "An admin task has priority. Your task will resume after admin task completes."
                    )
                except:
                    pass
        
        # Send success message
        status_text = "✅ **File added to queue!**\n\n"
        if is_admin:
            status_text += "🚨 **ADMIN PRIORITY**\n"
        
        status_text += f"📁 **File:** `{file_name[:50]}`\n"
        status_text += f"📦 **Size:** `{humanbytes(file_size)}`\n"
        status_text += f"👤 **User:** {'👑 Admin' if is_admin else '👤 User'}\n"
        status_text += f"🆔 **Task ID:** `{task_id}`\n"
        
        if queue_position == 1 and not processing_queue.is_processing:
            status_text += "🚀 **Starting processing immediately...**"
        else:
            wait_time = queue_position * 30
            status_text += f"📍 **Queue Position:** `{queue_position}`\n"
            status_text += f"⏳ **Estimated wait:** `{wait_time} seconds`\n"
            status_text += "⏳ **Files are processed one by one...**"
        
        await client.send_message(
            chat_id=chat_id,
            text=status_text,
            reply_to_message_id=message.id
        )
        
        print(f"✅ Added file to queue. Task ID: {task_id}, Position: {queue_position}, Admin: {is_admin}")
        
    except Exception as e:
        print(f"❌ Error in group file handler: {e}")
        import traceback
        traceback.print_exc()

# ==================== ADMIN COMMANDS ====================
@app.on_message(filters.command("stats") & filters.user(Config.ADMIN))
async def stats_handler(client, message):
    total_users = await db.total_users_count()
    uptime = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - Config.BOT_UPTIME))
    queue_info = processing_queue.get_queue_info()
    
    stats_text = f"**📊 Bot Statistics**\n\n"
    stats_text += f"**• Total Users:** `{total_users}`\n"
    stats_text += f"**• Uptime:** `{uptime}`\n"
    stats_text += f"**• Queue Status:** `{queue_info['total']} waiting, {queue_info['completed']} completed`\n"
    stats_text += f"**• Processing:** `{'Yes' if queue_info['is_processing'] else 'No'}`\n"
    stats_text += f"**• Failed Tasks:** `{queue_info['failed']}`\n"
    stats_text += f"**• Paused Tasks:** `{queue_info['paused']}`\n"
    stats_text += f"**• Admin Mode:** `{'Active' if queue_info['admin_mode'] else 'Inactive'}`\n"
    stats_text += f"**• Pyrogram Version:** `{__version__}`"
    
    await message.reply_text(stats_text)

@app.on_message(filters.command("clearqueue") & filters.user(Config.ADMIN))
async def clear_queue_handler(client, message):
    processing_queue.clear_queue()
    await message.reply_text("✅ **Queue cleared successfully!**")

@app.on_message(filters.command("restart") & filters.user(Config.ADMIN))
async def restart_handler(client, message):
    await message.reply_text("**🔄 Restarting bot...**")
    os.execl(sys.executable, sys.executable, *sys.argv)

# ==================== CALLBACK QUERY HANDLER ====================
@app.on_callback_query()
async def callback_handler(client, query):
    data = query.data
    user_id = query.from_user.id
    
    try:
        if data == "home":
            buttons = InlineKeyboardMarkup([
                [InlineKeyboardButton("📚 ʜᴇʟᴘ", callback_data='help'), InlineKeyboardButton("⚙️ ᴍᴇᴛᴀᴅᴀᴛᴀ", callback_data='metadata')],
                [
                    InlineKeyboardButton('📢 ᴜᴘᴅᴀᴛᴇs', url='https://t.me/Codeflix_Bots'),
                    InlineKeyboardButton('🆘 sᴜᴘᴘᴏʀᴛ', url='https://t.me/CodeflixSupport')
                ],
                [
                    InlineKeyboardButton('📊 Queue Status', callback_data='queue_status'),
                    InlineKeyboardButton('❌ ᴄʟᴏsᴇ', callback_data='close')
                ]
            ])
            
            await query.message.edit_text(
                Txt.START_TXT.format(query.from_user.mention),
                reply_markup=buttons,
                disable_web_page_preview=True
            )
            await query.answer()
        
        elif data == "help":
            buttons = InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 ʜᴏᴍᴇ", callback_data='home')],
                [InlineKeyboardButton("⚙️ ᴍᴇᴛᴀᴅᴀᴛᴀ", callback_data='metadata'), InlineKeyboardButton("📊 Queue", callback_data='queue_status')]
            ])
            
            await query.message.edit_text(
                Txt.HELP_TXT,
                reply_markup=buttons,
                disable_web_page_preview=True
            )
            await query.answer()
        
        elif data == "queue_status":
            await queue_status_handler(client, query.message)
            await query.answer("Queue status refreshed!")
        
        elif data == "refresh_queue":
            await queue_status_handler(client, query.message)
            await query.answer("Queue status refreshed!")
        
        elif data == "metadata":
            await ensure_user(user_id)
            metadata_status = await db.get_metadata(user_id)
            status_text = "ON ✅" if metadata_status else "OFF ❌"
            
            buttons = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Turn ON", callback_data="metadata_on"),
                    InlineKeyboardButton("❌ Turn OFF", callback_data="metadata_off")
                ],
                [InlineKeyboardButton("📊 View Settings", callback_data="view_metadata")],
                [InlineKeyboardButton("🏠 Home", callback_data="home")]
            ])
            
            await query.message.edit_text(
                f"**Metadata Status:** {status_text}\n\n"
                "Use buttons below to toggle metadata.",
                reply_markup=buttons
            )
            await query.answer()
        
        elif data == "cancel_my_tasks":
            cancelled_tasks = processing_queue.cancel_user_tasks(user_id)
            
            if cancelled_tasks:
                await query.answer(f"Cancelled {len(cancelled_tasks)} tasks!")
                await query.message.edit_text(f"✅ **Cancelled {len(cancelled_tasks)} of your tasks from the queue!**")
            else:
                await query.answer("You don't have any tasks to cancel!", show_alert=True)
        
        elif data == "close":
            try:
                await query.message.delete()
            except:
                await query.answer("Message deleted!", show_alert=True)
        
        else:
            await query.answer("Feature not implemented yet!", show_alert=True)
    
    except Exception as e:
        print(f"Callback error: {e}")
        try:
            await query.answer("Error processing request!", show_alert=True)
        except:
            pass

# ==================== OTHER COMMANDS ====================
@app.on_message(filters.command("set_caption"))
async def set_caption_handler(client, message):
    user_id = message.from_user.id
    await ensure_user(user_id)
    
    if len(message.command) < 2:
        await message.reply_text(
            "**Please provide a caption!**\n\n"
            "**Example:** `/set_caption File: {filename}\nSize: {filesize}\nDuration: {duration}`\n\n"
            "**Available variables:**\n"
            "- `{filename}`: File name\n"
            "- `{filesize}`: File size\n"
            "- `{duration}`: Duration\n\n"
            "**Note:** Use `\\n` for new line in caption."
        )
        return
    
    caption = message.text.split(" ", 1)[1]
    await db.set_caption(user_id, caption)
    await message.reply_text("✅ Caption set successfully!")

@app.on_message(filters.command("thumbnail"))
async def thumbnail_handler(client, message):
    user_id = message.from_user.id
    await ensure_user(user_id)
    
    if not message.reply_to_message or not message.reply_to_message.photo:
        await message.reply_text(
            "**Please reply to a photo with /thumbnail to set it as your thumbnail.**\n\n"
            "**Example:** Reply to a photo message with `/thumbnail`"
        )
        return
    
    photo = message.reply_to_message.photo
    await db.set_thumbnail(user_id, photo.file_id)
    await message.reply_text("✅ **Thumbnail saved successfully!**")

# ==================== MAIN ====================
async def main():
    """Main function to start the bot"""
    # Start the bot
    await app.start()
    
    # Start queue worker as background task
    asyncio.create_task(queue_worker())
    
    # Get bot info
    me = await app.get_me()
    print(f"✅ Bot started as @{me.username}")
    print(f"✅ Bot ID: {me.id}")
    print("✅ Admin Interruption System: ACTIVE")
    print("✅ When admin sends file, current task pauses")
    print("✅ Admin task processes immediately")
    print("✅ After admin task, paused task resumes")
    print("✅ Users get paused/resumed notifications")
    print("✅ Users can only cancel their own tasks")
    print("✅ Queue system with priority handling")
    
    # Send startup message to log channel
    try:
        await app.send_message(
            Config.LOG_CHANNEL,
            f"🤖 **Bot Started Successfully!**\n\n"
            f"**Name:** {me.first_name}\n"
            f"**Username:** @{me.username}\n"
            f"**ID:** `{me.id}`\n"
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Features:** Admin Interruption System Active"
        )
    except:
        pass
    
    # Keep the bot running
    await idle()
    
    # Stop the bot
    await app.stop()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check for ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print("✅ FFmpeg is installed and working")
        print(f"FFmpeg version: {result.stdout.split('version')[1].split(' ')[1] if 'version' in result.stdout else 'N/A'}")
    except:
        print("⚠️ WARNING: FFmpeg not found! Metadata features will not work.")
        print("Install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
    
    print("\n" + "="*60)
    print("🚀 Starting Auto Rename Bot with Admin Interruption System...")
    print("="*60)
    print(f"🤖 Bot Name: {Config.BOT_TOKEN.split(':')[0]}")
    print(f"👑 Admins: {Config.ADMIN}")
    print("📋 Queue System: ACTIVE")
    print("🚨 Admin Interruption: ENABLED")
    print("⏸️ Task Pausing: ENABLED")
    print("🔄 Task Resuming: ENABLED")
    print("📢 User Notifications: ENABLED")
    print("❌ User Task Control: ENABLED (Users can cancel own tasks)")
    print("⛔ Admin Only Control: Queue clearing, restart")
    print("📢 IMPORTANT: Add bot to your group and give it admin rights!")
    print("🤖 Bot is running. Press Ctrl+C to stop.")
    print("="*60 + "\n")
    
    try:
        # Run the bot
        app.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()