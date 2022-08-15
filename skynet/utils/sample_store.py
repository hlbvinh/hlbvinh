import pymongo


class SampleStore:
    def __init__(
        self,
        client,
        sample_collection,
        sample_id,
        index_order,
        watermark_collection,
        watermark_key,
        watermark_value,
        extra_indices=None,
    ):
        self.client = client
        self.sample_collection = sample_collection
        self.sample_id = sample_id
        self.watermark_collection = watermark_collection
        self.watermark_key = watermark_key
        self.watermark_value = watermark_value
        self.sort_order = list(zip(sample_id, index_order))
        self.client.create_index(sample_collection, self.sort_order)
        self.extra_indices = extra_indices
        if self.extra_indices is not None:
            self.add_indices(extra_indices)

    def add_indices(self, indices):
        for fields, orders in indices:
            sort_order = list(zip(fields, orders))
            self.client.create_index(self.sample_collection, sort_order)

    def get_watermark(self, key):
        latest_record = self.client.get_one(
            self.watermark_collection, {self.watermark_key: key}
        )
        return latest_record.get(self.watermark_value) if latest_record else None

    def get_watermarks(self):
        wms = self.client.get(self.watermark_collection)
        for wm in wms:
            wm.pop("_id", None)
        return wms

    def set_watermark(self, key, value):
        self.client.upsert(
            self.watermark_collection,
            {self.watermark_key: key, self.watermark_value: value},
            key={self.watermark_key: key},
        )

    def upsert(self, sample):
        try:
            key = {k: sample[k] for k in self.sample_id}
        except KeyError:
            raise ValueError("sample missing id keys")

        return self.client.upsert(self.sample_collection, sample, key)

    def upsert_many(self, samples):
        keys = []
        for sample in samples:
            try:
                keys.append({k: sample[k] for k in self.sample_id})
            except KeyError:
                raise ValueError("sample missing id keys")

        self.client.upsert_many(self.sample_collection, samples, keys)

    def get(self, key={}, sort=None, limit=0, direction=pymongo.ASCENDING):
        records = self.client.get(self.sample_collection, key, sort, limit, direction)
        for rec in records:
            rec.pop("_id", None)
        return records

    def clear(self, key={}):
        self.client.remove(self.sample_collection, key)

    def reset_watermarks(self, key={}):
        self.client.remove(self.watermark_collection, key)

    def clear_all(self):
        self.clear()
        self.reset_watermarks()
