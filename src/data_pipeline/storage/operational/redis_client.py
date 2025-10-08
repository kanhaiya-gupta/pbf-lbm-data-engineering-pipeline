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

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis client for caching and session management operations in the operational layer.
    
    Handles key-value operations, caching, pub/sub, and real-time data
    management for PBF-LB/M operational systems.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 password: Optional[str] = None, db: int = 0,
                 max_connections: int = 50):
        """
        Initialize Redis client.
        
        Args:
            host: Redis server host
            port: Redis server port
            password: Redis password
            db: Redis database number
            max_connections: Maximum number of connections in pool
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.max_connections = max_connections
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        
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
            
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def disconnect(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()
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
            logger.error(f"Failed to get key {key}: {e}")
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
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:hash:{key_hash}"
        
        return key_string
