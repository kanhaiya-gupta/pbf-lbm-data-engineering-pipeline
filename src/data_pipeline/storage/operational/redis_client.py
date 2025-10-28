"""
Redis Client for Caching and Session Management in Operational Layer

This module provides Redis integration for high-performance caching,
session management, and real-time data operations in the operational layer.
Particularly useful for PBF-LB/M process state management, real-time
monitoring data caching, and user session handling.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import json
import pickle
from redis import Redis, ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError
import hashlib
import uuid

# Import our Redis configuration
from src.data_pipeline.config.redis_config import RedisConfig, get_redis_config

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis client for caching and session management operations in the operational layer.
    
    Handles key-value operations, caching, pub/sub, and real-time data
    management for PBF-LB/M operational systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, host: str = 'localhost', port: int = 6379, 
                 password: Optional[str] = None, db: int = 0,
                 max_connections: int = 50):
        """
        Initialize Redis client.
        
        Args:
            config: Redis configuration dictionary (preferred)
            host: Redis server host (fallback if no config)
            port: Redis server port (fallback if no config)
            password: Redis password (fallback if no config)
            db: Redis database number (fallback if no config)
            max_connections: Maximum number of connections in pool
        """
        # Use config if provided, otherwise use individual parameters
        if config:
            self.host = config.get('host', 'localhost')
            self.port = config.get('port', 6379)
            self.password = config.get('password')
            self.db = config.get('db', 0)
            self.max_connections = config.get('max_connections', 50)
            self.default_ttl = config.get('default_ttl', 3600)
            self.process_cache_ttl = config.get('process_cache_ttl', 3600)
            self.session_ttl = config.get('session_ttl', 3600)
            self.analytics_ttl = config.get('analytics_ttl', 3600)
        else:
            self.host = host
            self.port = port
            self.password = password
            self.db = db
            self.max_connections = max_connections
            self.default_ttl = 3600  # 1 hour default
            self.process_cache_ttl = 1800  # 30 minutes
            self.session_ttl = 3600  # 1 hour
            self.analytics_ttl = 86400  # 24 hours
            self._pool: Optional[ConnectionPool] = None
            self._client: Optional[Redis] = None
            self._connected: bool = False
        
    def connect(self) -> bool:
        """
        Establish connection to Redis.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            self._client = Redis(connection_pool=self._pool)
            
            # Test connection
            self._client.ping()
            self._connected = True
            
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def disconnect(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("Disconnected from Redis")
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        Set a key-value pair.
        
        Args:
            key: Key name
            value: Value to store
            expire: Expiration time in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            # Serialize value
            serialized_value = self._serialize(value)
            
            result = self._client.set(key, serialized_value, ex=expire)
            logger.debug(f"Set key: {key}")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to set key {key}: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value by key.
        
        Args:
            key: Key name
            
        Returns:
            Optional[Any]: Value if found, None otherwise
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            value = self._client.get(key)
            if value is None:
                return None
            
            return self._deserialize(value)
            
        except RedisError as e:
            logger.debug(f"Failed to get key {key}: {e}")
            raise
    
    def delete(self, *keys: str) -> int:
        """
        Delete one or more keys.
        
        Args:
            *keys: Keys to delete
            
        Returns:
            int: Number of keys deleted
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            result = self._client.delete(*keys)
            logger.debug(f"Deleted {result} keys")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to delete keys: {e}")
            raise
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: Key name
            
        Returns:
            bool: True if key exists
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            return bool(self._client.exists(key))
            
        except RedisError as e:
            logger.error(f"Failed to check key existence: {e}")
            raise
    
    def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Key name
            seconds: Expiration time in seconds
            
        Returns:
            bool: True if expiration was set
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            return bool(self._client.expire(key, seconds))
            
        except RedisError as e:
            logger.error(f"Failed to set expiration for key {key}: {e}")
            raise
    
    def ttl(self, key: str) -> int:
        """
        Get time to live for a key.
        
        Args:
            key: Key name
            
        Returns:
            int: TTL in seconds (-1 if no expiration, -2 if key doesn't exist)
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            return self._client.ttl(key)
            
        except RedisError as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            raise
    
    def mset(self, mapping: Dict[str, Any]) -> bool:
        """
        Set multiple key-value pairs.
        
        Args:
            mapping: Dictionary of key-value pairs
            
        Returns:
            bool: True if successful
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            # Serialize values
            serialized_mapping = {
                key: self._serialize(value) 
                for key, value in mapping.items()
            }
            
            result = self._client.mset(serialized_mapping)
            logger.debug(f"Set {len(mapping)} keys")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to mset keys: {e}")
            raise
    
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """
        Get multiple values by keys.
        
        Args:
            keys: List of key names
            
        Returns:
            List[Optional[Any]]: List of values
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            values = self._client.mget(keys)
            return [self._deserialize(v) if v is not None else None for v in values]
            
        except RedisError as e:
            logger.error(f"Failed to mget keys: {e}")
            raise
    
    def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """
        Set hash field values.
        
        Args:
            name: Hash name
            mapping: Dictionary of field-value pairs
            
        Returns:
            int: Number of fields set
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            # Serialize values
            serialized_mapping = {
                field: self._serialize(value) 
                for field, value in mapping.items()
            }
            
            result = self._client.hset(name, mapping=serialized_mapping)
            logger.debug(f"Set {result} hash fields in {name}")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to hset hash {name}: {e}")
            raise
    
    def hget(self, name: str, key: str) -> Optional[Any]:
        """
        Get hash field value.
        
        Args:
            name: Hash name
            key: Field name
            
        Returns:
            Optional[Any]: Field value if found, None otherwise
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            value = self._client.hget(name, key)
            if value is None:
                return None
            
            return self._deserialize(value)
            
        except RedisError as e:
            logger.error(f"Failed to hget field {key} from hash {name}: {e}")
            raise
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """
        Get all hash field values.
        
        Args:
            name: Hash name
            
        Returns:
            Dict[str, Any]: Dictionary of field-value pairs
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            hash_data = self._client.hgetall(name)
            return {
                field.decode('utf-8'): self._deserialize(value)
                for field, value in hash_data.items()
            }
            
        except RedisError as e:
            logger.error(f"Failed to hgetall hash {name}: {e}")
            raise
    
    def lpush(self, name: str, *values: Any) -> int:
        """
        Push values to the left of a list.
        
        Args:
            name: List name
            *values: Values to push
            
        Returns:
            int: Length of list after push
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            # Serialize values
            serialized_values = [self._serialize(v) for v in values]
            
            result = self._client.lpush(name, *serialized_values)
            logger.debug(f"Pushed {len(values)} values to list {name}")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to lpush to list {name}: {e}")
            raise
    
    def rpop(self, name: str) -> Optional[Any]:
        """
        Pop value from the right of a list.
        
        Args:
            name: List name
            
        Returns:
            Optional[Any]: Popped value if list not empty, None otherwise
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            value = self._client.rpop(name)
            if value is None:
                return None
            
            return self._deserialize(value)
            
        except RedisError as e:
            logger.error(f"Failed to rpop from list {name}: {e}")
            raise
    
    def llen(self, name: str) -> int:
        """
        Get list length.
        
        Args:
            name: List name
            
        Returns:
            int: List length
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            return self._client.llen(name)
            
        except RedisError as e:
            logger.error(f"Failed to get length of list {name}: {e}")
            raise
    
    def sadd(self, name: str, *values: Any) -> int:
        """
        Add values to a set.
        
        Args:
            name: Set name
            *values: Values to add
            
        Returns:
            int: Number of values added
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            # Serialize values
            serialized_values = [self._serialize(v) for v in values]
            
            result = self._client.sadd(name, *serialized_values)
            logger.debug(f"Added {result} values to set {name}")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to sadd to set {name}: {e}")
            raise
    
    def smembers(self, name: str) -> set:
        """
        Get all set members.
        
        Args:
            name: Set name
            
        Returns:
            set: Set of members
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            members = self._client.smembers(name)
            return {self._deserialize(m) for m in members}
            
        except RedisError as e:
            logger.error(f"Failed to get members of set {name}: {e}")
            raise
    
    def zadd(self, name: str, mapping: Dict[Any, float]) -> int:
        """
        Add values to a sorted set.
        
        Args:
            name: Sorted set name
            mapping: Dictionary of value-score pairs
            
        Returns:
            int: Number of values added
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            # Serialize values
            serialized_mapping = {
                self._serialize(value): score 
                for value, score in mapping.items()
            }
            
            result = self._client.zadd(name, serialized_mapping)
            logger.debug(f"Added {result} values to sorted set {name}")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to zadd to sorted set {name}: {e}")
            raise
    
    def zrange(self, name: str, start: int, end: int, 
               withscores: bool = False) -> List[Any]:
        """
        Get range of values from sorted set.
        
        Args:
            name: Sorted set name
            start: Start index
            end: End index
            withscores: Whether to include scores
            
        Returns:
            List[Any]: List of values (with scores if requested)
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            values = self._client.zrange(name, start, end, withscores=withscores)
            
            if withscores:
                return [(self._deserialize(v), s) for v, s in values]
            else:
                return [self._deserialize(v) for v in values]
            
        except RedisError as e:
            logger.error(f"Failed to zrange sorted set {name}: {e}")
            raise
    
    def publish(self, channel: str, message: Any) -> int:
        """
        Publish message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            
        Returns:
            int: Number of subscribers that received the message
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            serialized_message = self._serialize(message)
            result = self._client.publish(channel, serialized_message)
            logger.debug(f"Published message to channel {channel}")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            raise
    
    def subscribe(self, *channels: str):
        """
        Subscribe to channels.
        
        Args:
            *channels: Channel names to subscribe to
            
        Returns:
            PubSub: PubSub object for receiving messages
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            pubsub = self._client.pubsub()
            pubsub.subscribe(*channels)
            logger.info(f"Subscribed to channels: {channels}")
            return pubsub
            
        except RedisError as e:
            logger.error(f"Failed to subscribe to channels: {e}")
            raise
    
    def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set a cached value with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        return self.set(key, value, expire=ttl)
    
    def cache_get(self, key: str) -> Optional[Any]:
        """
        Get a cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value if found and not expired
        """
        return self.get(key)
    
    def cache_delete(self, pattern: str) -> int:
        """
        Delete cache keys matching pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            int: Number of keys deleted
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            keys = self._client.keys(pattern)
            if keys:
                return self.delete(*keys)
            return 0
            
        except RedisError as e:
            logger.error(f"Failed to delete cache keys with pattern {pattern}: {e}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict: Cache statistics
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            info = self._client.info('memory')
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'maxmemory': info.get('maxmemory', 0),
                'maxmemory_human': info.get('maxmemory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }
            
        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e}")
            raise
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            bytes: Serialized value
        """
        if isinstance(value, (str, int, float, bool)):
            return str(value).encode('utf-8')
        elif isinstance(value, (dict, list)):
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps(value)
    
    def _deserialize(self, value: bytes) -> Any:
        """
        Deserialize value from storage.
        
        Args:
            value: Serialized value
            
        Returns:
            Any: Deserialized value
        """
        try:
            # Try JSON first
            return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Try pickle
                return pickle.loads(value)
            except (pickle.PickleError, UnicodeDecodeError):
                # Return as string
                return value.decode('utf-8')
    
    def generate_cache_key(self, prefix: str, *args: Any) -> str:
        """
        Generate a cache key from prefix and arguments.
        
        Args:
            prefix: Key prefix
            *args: Arguments to include in key
            
        Returns:
            str: Generated cache key
        """
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = ':'.join(key_parts)
        
        # Hash long keys to avoid Redis key length limits
        if len(key_string) > 250:
            key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
            return f"{prefix}:hash:{key_hash}"
        
        return key_string
    
    # PBF-LB/M Specific Caching Methods
    
    def cache_process_data(self, process_id: str, process_data: Dict[str, Any]) -> bool:
        """
        Cache process data with appropriate TTL.
        
        Args:
            process_id: Process identifier
            process_data: Process data to cache
            
        Returns:
            bool: True if successful
        """
        key = f"process:{process_id}"
        return self.set_hash(key, process_data, ttl=self.process_cache_ttl)
    
    def get_process_data(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached process data.
        
        Args:
            process_id: Process identifier
            
        Returns:
            Optional[Dict]: Cached process data or None
        """
        key = f"process:{process_id}"
        return self.get_hash(key)
    
    def cache_machine_status(self, machine_id: str, status_data: Dict[str, Any]) -> bool:
        """
        Cache machine status data.
        
        Args:
            machine_id: Machine identifier
            status_data: Machine status data
            
        Returns:
            bool: True if successful
        """
        key = f"machine:{machine_id}:status"
        return self.set_hash(key, status_data, ttl=self.default_ttl)
    
    def get_machine_status(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached machine status.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            Optional[Dict]: Cached machine status or None
        """
        key = f"machine:{machine_id}:status"
        return self.get_hash(key)
    
    def cache_sensor_reading(self, sensor_id: str, reading_data: Dict[str, Any]) -> bool:
        """
        Cache latest sensor reading.
        
        Args:
            sensor_id: Sensor identifier
            reading_data: Sensor reading data
            
        Returns:
            bool: True if successful
        """
        key = f"sensor:{sensor_id}:latest"
        return self.set_hash(key, reading_data, ttl=300)  # 5 minutes TTL for sensor data
    
    def get_sensor_reading(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached sensor reading.
        
        Args:
            sensor_id: Sensor identifier
            
        Returns:
            Optional[Dict]: Cached sensor reading or None
        """
        key = f"sensor:{sensor_id}:latest"
        return self.get_hash(key)
    
    def cache_analytics(self, analytics_type: str, date: str, analytics_data: Dict[str, Any]) -> bool:
        """
        Cache analytics data.
        
        Args:
            analytics_type: Type of analytics (daily, weekly, monthly)
            date: Date string
            analytics_data: Analytics data
            
        Returns:
            bool: True if successful
        """
        key = f"analytics:{analytics_type}:{date}"
        return self.set_hash(key, analytics_data, ttl=self.analytics_ttl)
    
    def get_analytics(self, analytics_type: str, date: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analytics data.
        
        Args:
            analytics_type: Type of analytics
            date: Date string
            
        Returns:
            Optional[Dict]: Cached analytics data or None
        """
        key = f"analytics:{analytics_type}:{date}"
        return self.get_hash(key)
    
    def add_to_job_queue(self, job_type: str, job_data: Dict[str, Any]) -> bool:
        """
        Add job to processing queue.
        
        Args:
            job_type: Type of job
            job_data: Job data
            
        Returns:
            bool: True if successful
        """
        job = {
            "type": job_type,
            "data": job_data,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        return self.push_to_list("job:queue", job)
    
    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Get next job from processing queue.
        
        Returns:
            Optional[Dict]: Next job or None if queue is empty
        """
        return self.pop_from_list("job:queue")
    
    def cache_user_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """
        Cache user session data.
        
        Args:
            user_id: User identifier
            session_data: Session data
            
        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        key = f"user:{user_id}:session:{session_id}"
        self.set_hash(key, session_data, ttl=self.session_ttl)
        return session_id
    
    def get_user_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached user session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Optional[Dict]: Session data or None
        """
        key = f"user:{user_id}:session:{session_id}"
        return self.get_hash(key)
    
    def invalidate_process_cache(self, process_id: str) -> bool:
        """
        Invalidate process cache.
        
        Args:
            process_id: Process identifier
            
        Returns:
            bool: True if successful
        """
        key = f"process:{process_id}"
        return self.delete(key)
    
    def invalidate_machine_cache(self, machine_id: str) -> bool:
        """
        Invalidate machine cache.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            bool: True if successful
        """
        pattern = f"machine:{machine_id}:*"
        keys = self._get_keys(pattern)
        if keys:
            return self.delete(*keys) > 0
        return True
    
    # Missing helper methods that are referenced in PBF-specific methods
    
    def set_hash(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set hash data with optional TTL.
        
        Args:
            key: Hash key
            data: Hash data
            ttl: Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            # Set hash data
            result = self.hset(key, data)
            
            # Set TTL if provided
            if ttl and result >= 0:  # Allow 0 result (no new fields added)
                self.expire(key, ttl)
            
            # Consider success if hset completed without error (even if 0 fields added)
            return result >= 0
            
        except Exception as e:
            logger.error(f"Failed to set hash {key}: {e}")
            return False
    
    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get hash data.
        
        Args:
            key: Hash key
            
        Returns:
            Optional[Dict]: Hash data or None
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            return self.hgetall(key)
            
        except Exception as e:
            logger.error(f"Failed to get hash {key}: {e}")
            return None
    
    def push_to_list(self, key: str, value: Any) -> bool:
        """
        Push value to list.
        
        Args:
            key: List key
            value: Value to push
            
        Returns:
            bool: True if successful
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            result = self.lpush(key, value)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to push to list {key}: {e}")
            return False
    
    def pop_from_list(self, key: str) -> Optional[Any]:
        """
        Pop value from list.
        
        Args:
            key: List key
            
        Returns:
            Optional[Any]: Popped value or None
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            return self.rpop(key)
            
        except Exception as e:
            logger.error(f"Failed to pop from list {key}: {e}")
            return None
    
    def get_list_length(self, key: str) -> int:
        """
        Get list length.
        
        Args:
            key: List key
            
        Returns:
            int: List length
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            return self.llen(key)
            
        except Exception as e:
            logger.error(f"Failed to get list length {key}: {e}")
            return 0
    
    def get_keys(self, pattern: str) -> List[str]:
        """
        Get keys matching pattern using SCAN (production-safe).
        
        Args:
            pattern: Key pattern
            
        Returns:
            List[str]: Matching keys
        """
        try:
            keys = []
            cursor = 0
            while True:
                cursor, partial_keys = self._client.scan(cursor, match=pattern, count=1000)
                keys.extend(partial_keys)
                if cursor == 0:
                    break
            return keys
        except Exception as e:
            logger.error(f"Failed to scan keys with pattern {pattern}: {e}")
            return []
    
    def _get_keys(self, pattern: str) -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List[str]: Matching keys
        """
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            keys = self._client.keys(pattern)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
            
        except Exception as e:
            logger.error(f"Failed to get keys with pattern {pattern}: {e}")
            return []
    
    def is_connected(self) -> bool:
        """
        Check if Redis connection is active.
        
        Returns:
            bool: True if connected
        """
        if not self._connected or not self._client:
            return False
        
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False
    
    def reconnect(self) -> bool:
        """
        Reconnect to Redis if disconnected.
        
        Returns:
            bool: True if reconnection successful
        """
        if self.is_connected():
            return True
        
        self.disconnect()
        return self.connect()
    
    def close(self):
        """Close Redis connection."""
        self.disconnect()
    
    def close_connection(self):
        """Close Redis connection (alias for close)."""
        self.close()
    
    def get_all_process_cache(self) -> List[Dict[str, Any]]:
        """Get all process cache data from Redis."""
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            keys = self.get_keys("process:*")
            data = []
            for key in keys:
                # Use hgetall for hash data
                hash_data = self._client.hgetall(key)
                if hash_data:
                    # Convert bytes keys and values to strings with error handling
                    processed_data = {}
                    for k, v in hash_data.items():
                        # Handle key decoding
                        if isinstance(k, bytes):
                            try:
                                key_str = k.decode('utf-8')
                            except UnicodeDecodeError:
                                key_str = k.decode('utf-8', errors='replace')
                        else:
                            key_str = str(k)
                        
                        # Handle value decoding
                        if isinstance(v, bytes):
                            try:
                                val_str = v.decode('utf-8')
                            except UnicodeDecodeError:
                                # For binary data, convert to base64 or skip
                                val_str = f"<binary_data_{len(v)}_bytes>"
                        else:
                            val_str = str(v)
                        
                        processed_data[key_str] = val_str
                    data.append(processed_data)
            return data
        except Exception as e:
            logger.error(f"Failed to get process cache: {e}")
            return []
    
    def get_all_machine_status_cache(self) -> List[Dict[str, Any]]:
        """Get all machine status cache data from Redis."""
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            keys = self.get_keys("machine:*")
            data = []
            for key in keys:
                # Use hgetall for hash data
                hash_data = self._client.hgetall(key)
                if hash_data:
                    # Convert bytes keys and values to strings with error handling
                    processed_data = {}
                    for k, v in hash_data.items():
                        # Handle key decoding
                        if isinstance(k, bytes):
                            try:
                                key_str = k.decode('utf-8')
                            except UnicodeDecodeError:
                                key_str = k.decode('utf-8', errors='replace')
                        else:
                            key_str = str(k)
                        
                        # Handle value decoding
                        if isinstance(v, bytes):
                            try:
                                val_str = v.decode('utf-8')
                            except UnicodeDecodeError:
                                # For binary data, convert to base64 or skip
                                val_str = f"<binary_data_{len(v)}_bytes>"
                        else:
                            val_str = str(v)
                        
                        processed_data[key_str] = val_str
                    data.append(processed_data)
            return data
        except Exception as e:
            logger.error(f"Failed to get machine status cache: {e}")
            return []
    
    def get_all_sensor_readings_cache(self) -> List[Dict[str, Any]]:
        """Get all sensor readings cache data from Redis."""
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            keys = self.get_keys("sensor:*")
            data = []
            for key in keys:
                # Use hgetall for hash data
                hash_data = self._client.hgetall(key)
                if hash_data:
                    # Convert bytes keys and values to strings with error handling
                    processed_data = {}
                    for k, v in hash_data.items():
                        # Handle key decoding
                        if isinstance(k, bytes):
                            try:
                                key_str = k.decode('utf-8')
                            except UnicodeDecodeError:
                                key_str = k.decode('utf-8', errors='replace')
                        else:
                            key_str = str(k)
                        
                        # Handle value decoding
                        if isinstance(v, bytes):
                            try:
                                val_str = v.decode('utf-8')
                            except UnicodeDecodeError:
                                # For binary data, convert to base64 or skip
                                val_str = f"<binary_data_{len(v)}_bytes>"
                        else:
                            val_str = str(v)
                        
                        processed_data[key_str] = val_str
                    data.append(processed_data)
            return data
        except Exception as e:
            logger.error(f"Failed to get sensor readings cache: {e}")
            return []
    
    def get_all_analytics_cache(self) -> List[Dict[str, Any]]:
        """Get all analytics cache data from Redis."""
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            keys = self.get_keys("analytics:*")
            data = []
            for key in keys:
                # Use hgetall for hash data
                hash_data = self._client.hgetall(key)
                if hash_data:
                    # Convert bytes keys and values to strings with error handling
                    processed_data = {}
                    for k, v in hash_data.items():
                        # Handle key decoding
                        if isinstance(k, bytes):
                            try:
                                key_str = k.decode('utf-8')
                            except UnicodeDecodeError:
                                key_str = k.decode('utf-8', errors='replace')
                        else:
                            key_str = str(k)
                        
                        # Handle value decoding
                        if isinstance(v, bytes):
                            try:
                                val_str = v.decode('utf-8')
                            except UnicodeDecodeError:
                                # For binary data, convert to base64 or skip
                                val_str = f"<binary_data_{len(v)}_bytes>"
                        else:
                            val_str = str(v)
                        
                        processed_data[key_str] = val_str
                    data.append(processed_data)
            return data
        except Exception as e:
            logger.error(f"Failed to get analytics cache: {e}")
            return []
    
    def get_all_job_queue_items(self) -> List[Dict[str, Any]]:
        """Get all job queue items from Redis."""
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            keys = self.get_keys("job:*")
            data = []
            for key in keys:
                # Use hgetall for hash data
                hash_data = self._client.hgetall(key)
                if hash_data:
                    # Convert bytes keys and values to strings with error handling
                    processed_data = {}
                    for k, v in hash_data.items():
                        # Handle key decoding
                        if isinstance(k, bytes):
                            try:
                                key_str = k.decode('utf-8')
                            except UnicodeDecodeError:
                                key_str = k.decode('utf-8', errors='replace')
                        else:
                            key_str = str(k)
                        
                        # Handle value decoding
                        if isinstance(v, bytes):
                            try:
                                val_str = v.decode('utf-8')
                            except UnicodeDecodeError:
                                # For timestamp fields, set to None instead of binary data string
                                if key_str.lower() in ['scheduled_at', 'last_activity', 'created_at', 'updated_at', 'completed_at', 'expires_at']:
                                    val_str = None
                                else:
                                    val_str = f"<binary_data_{len(v)}_bytes>"
                        else:
                            val_str = str(v)
                        
                        processed_data[key_str] = val_str
                    data.append(processed_data)
            return data
        except Exception as e:
            logger.error(f"Failed to get job queue items: {e}")
            return []
    
    def get_all_user_sessions(self) -> List[Dict[str, Any]]:
        """Get all user sessions from Redis."""
        try:
            if not self._client:
                raise RuntimeError("Not connected to Redis")
            
            keys = self.get_keys("session:*")
            data = []
            for key in keys:
                # Use hgetall for hash data
                hash_data = self._client.hgetall(key)
                if hash_data:
                    # Convert bytes keys and values to strings with error handling
                    processed_data = {}
                    for k, v in hash_data.items():
                        # Handle key decoding
                        if isinstance(k, bytes):
                            try:
                                key_str = k.decode('utf-8')
                            except UnicodeDecodeError:
                                key_str = k.decode('utf-8', errors='replace')
                        else:
                            key_str = str(k)
                        
                        # Handle value decoding
                        if isinstance(v, bytes):
                            try:
                                val_str = v.decode('utf-8')
                            except UnicodeDecodeError:
                                # For timestamp fields, set to None instead of binary data string
                                if key_str.lower() in ['scheduled_at', 'last_activity', 'created_at', 'updated_at', 'completed_at', 'expires_at']:
                                    val_str = None
                                else:
                                    val_str = f"<binary_data_{len(v)}_bytes>"
                        else:
                            val_str = str(v)
                        
                        processed_data[key_str] = val_str
                    data.append(processed_data)
            return data
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    def load_data(
        self,
        df: Any,
        key_prefix: str,
        mode: str = "append",
        batch_size: int = 1000,
        ttl: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load Spark DataFrame data into Redis cache.
        
        This method provides ETL pipeline integration for loading transformed
        Spark DataFrames into Redis, following the ETL architecture.
        
        Args:
            df: Spark DataFrame from transform modules
            key_prefix: Prefix for Redis keys (e.g., "cache:", "session:")
            mode: Write mode (append, overwrite, ignore, error)
            batch_size: Batch size for processing
            ttl: Time-to-live in seconds for keys (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Loading results and statistics
        """
        try:
            logger.info(f"Loading Spark DataFrame into Redis with prefix: {key_prefix}")
            
            # Initialize result tracking
            result = {
                "success": False,
                "keys_loaded": 0,
                "keys_processed": 0,
                "errors": [],
                "warnings": [],
                "key_prefix": key_prefix,
                "mode": mode,
                "batch_size": batch_size,
                "ttl": ttl
            }
            
            # Convert Spark DataFrame to list of dictionaries
            data_list = self._convert_spark_dataframe(df)
            if not data_list:
                result["warnings"].append("No data to load")
                return result
            
            result["keys_processed"] = len(data_list)
            logger.info(f"Converted {len(data_list)} records from Spark DataFrame")
            
            # Handle different modes
            if mode == "overwrite":
                # Clear existing keys with prefix
                existing_keys = self.get_keys(f"{key_prefix}*")
                if existing_keys:
                    self._client.delete(*existing_keys)
                    logger.info(f"Cleared {len(existing_keys)} existing keys with prefix: {key_prefix}")
            
            # Batch processing
            total_loaded = 0
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                try:
                    # Process batch
                    batch_result = self._process_redis_batch(key_prefix, batch, ttl)
                    
                    total_loaded += batch_result["loaded"]
                    
                    if batch_result["errors"]:
                        result["errors"].extend(batch_result["errors"])
                    
                    logger.info(f"Processed batch {i//batch_size + 1}: {batch_result['loaded']} keys")
                    
                except Exception as e:
                    error_msg = f"Error processing batch {i//batch_size + 1}: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Update result
            result["keys_loaded"] = total_loaded
            result["success"] = total_loaded > 0 and len(result["errors"]) == 0
            
            if result["success"]:
                logger.info(f"Successfully loaded {total_loaded} keys into Redis with prefix: {key_prefix}")
            else:
                logger.error(f"Failed to load data into Redis. Errors: {result['errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in load_data for prefix {key_prefix}: {str(e)}")
            return {
                "success": False,
                "keys_loaded": 0,
                "keys_processed": 0,
                "errors": [str(e)],
                "warnings": [],
                "key_prefix": key_prefix,
                "mode": mode,
                "batch_size": batch_size,
                "ttl": ttl
            }
    
    def _convert_spark_dataframe(self, df: Any) -> List[Dict[str, Any]]:
        """
        Convert Spark DataFrame to list of dictionaries for Redis insertion.
        
        Args:
            df: Spark DataFrame from transform modules
            
        Returns:
            List[Dict[str, Any]]: Converted data list
        """
        try:
            if hasattr(df, 'collect'):
                # Spark DataFrame - convert to list of dicts
                rows = df.collect()
                data_list = []
                
                for row in rows:
                    # Convert Row to dictionary
                    row_dict = row.asDict()
                    
                    # Handle Spark-specific data types
                    processed_dict = self._process_spark_row(row_dict)
                    data_list.append(processed_dict)
                
                return data_list
                
            elif isinstance(df, list):
                # Already a list of dictionaries
                return df
                
            elif isinstance(df, dict):
                # Single dictionary
                return [df]
                
            else:
                logger.warning(f"Unsupported DataFrame type: {type(df)}")
                return []
                
        except Exception as e:
            logger.error(f"Error converting Spark DataFrame: {str(e)}")
            return []
    
    def _process_spark_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Spark Row data to Redis-compatible format.
        
        Args:
            row_dict: Dictionary from Spark Row
            
        Returns:
            Dict[str, Any]: Processed dictionary
        """
        try:
            processed = {}
            
            for key, value in row_dict.items():
                # Handle None values
                if value is None:
                    processed[key] = None
                # Handle Spark-specific types
                elif hasattr(value, 'isoformat'):
                    # Datetime objects - convert to ISO string
                    processed[key] = value.isoformat()
                elif isinstance(value, (int, float, str, bool)):
                    # Basic types
                    processed[key] = value
                elif isinstance(value, dict):
                    # Nested dictionaries - convert to JSON string
                    processed[key] = json.dumps(value)
                elif isinstance(value, list):
                    # Lists - convert to JSON string
                    processed[key] = json.dumps(value)
                elif hasattr(value, '__dict__'):
                    # Complex objects - convert to JSON string
                    processed[key] = json.dumps(value.__dict__, default=str)
                else:
                    # Fallback - convert to string
                    processed[key] = str(value)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing Spark row: {str(e)}")
            return row_dict  # Return original if processing fails
    
    def _process_redis_batch(
        self, 
        key_prefix: str, 
        batch: List[Dict[str, Any]], 
        ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of records for Redis insertion.
        
        Args:
            key_prefix: Prefix for Redis keys
            batch: List of data dictionaries
            ttl: Time-to-live in seconds
            
        Returns:
            Dict[str, Any]: Batch processing results
        """
        try:
            loaded = 0
            errors = []
            
            for data in batch:
                try:
                    # Generate unique key for this record
                    if 'id' in data:
                        key = f"{key_prefix}{data['id']}"
                    elif 'key' in data:
                        key = f"{key_prefix}{data['key']}"
                    else:
                        # Generate UUID-based key
                        key = f"{key_prefix}{uuid.uuid4().hex}"
                    
                    # Determine data structure based on content
                    if len(data) == 1:
                        # Single value - use string
                        value = list(data.values())[0]
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        self.set_value(key, value, ttl=ttl)
                    else:
                        # Multiple values - use hash
                        self.set_hash(key, data, ttl=ttl)
                    
                    loaded += 1
                    
                except Exception as e:
                    error_msg = f"Failed to load record: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
            
            return {
                "loaded": loaded,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error processing Redis batch: {str(e)}")
            return {
                "loaded": 0,
                "errors": [str(e)]
            }